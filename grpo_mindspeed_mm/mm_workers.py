import logging
import os
import re
import shutil
import warnings

import torch
from omegaconf import DictConfig

try:
    # for torch 2.5+
    from torch.distributed.tensor import DTensor
except ImportError:
    from torch.distributed._tensor import DTensor

from verl.models.transformers.monkey_patch import apply_monkey_patch
from verl.single_controller.base.decorator import Dispatch, register
from verl.utils import hf_processor, hf_tokenizer
from verl.utils.device import (
    get_device_id,
    set_expandable_segments,
)
from verl.utils.fsdp_utils import (
    collect_lora_params,
    load_fsdp_model_to_gpu,
    offload_fsdp_model_to_cpu,
    offload_fsdp_optimizer,
    replace_lora_wrapper,
)
from verl.utils.memory_utils import aggressive_empty_cache
from verl.utils.model import convert_weight_keys
from verl.utils.profiler import log_gpu_memory_usage
from verl.workers.config import FSDPEngineConfig
from verl.workers.fsdp_workers import ActorRolloutRefWorker, AsyncActorRolloutRefWorker, get_vl_model_vision_tower

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def move_buffers_to_device_recursive(model, device):
    def _move_buffers(t):
        if not isinstance(t, torch.nn.Parameter):
            return t.to(device)
        else:
            return t

    return model._apply(_move_buffers, recurse=True)


class MMActorRolloutRefWorker(ActorRolloutRefWorker):
    def __init__(self, config: DictConfig, role: str, **kwargs):
        super().__init__(config, role, **kwargs)
        self.trainer = None

    def _dataloader(self, args=None):
        return None

    async def rollout_mode(self):
        """Context switch hybridengine to rollout mode."""
        aggressive_empty_cache(force_sync=True)

        log_gpu_memory_usage("Before load_fsdp_model_to_gpu", logger=logger)
        if self._is_offload_param:
            load_fsdp_model_to_gpu(self.actor_module_fsdp)
        log_gpu_memory_usage("After load_fsdp_model_to_gpu", logger=logger)

        peft_config = None
        peft_model = getattr(self.actor_module_fsdp, "_fsdp_wrapped_module", self.actor_module_fsdp)
        if hasattr(peft_model, "peft_config"):
            peft_config = peft_model.peft_config.get("default", None)
            params = collect_lora_params(
                module=self.actor_module_fsdp,
                layered_summon=self.config.rollout.get("layered_summon", False),
                base_sync_done=self.base_sync_done,
            )
            if not self.base_sync_done:
                params = {replace_lora_wrapper(k, peft_config): v for k, v in params.items()}
        else:
            if self._is_offload_param:
                params = self.actor_module_fsdp.state_dict()
            else:
                move_buffers_to_device_recursive(self.actor_module_fsdp, "cpu")
                params = self.actor_module_fsdp.state_dict()
                move_buffers_to_device_recursive(self.actor_module_fsdp, "npu")

        params = convert_weight_keys(
            params, getattr(self.actor_module_fsdp, "_fsdp_wrapped_module", self.actor_module_fsdp)
        )

        for k in list(params.keys()):
            if "mlp.experts.gate_up_proj" in k or "mlp.experts.down_proj" in k:
                print(f"--> modify shape {k}")
                params[k] = params[k].transpose(1, 2).contiguous()

        # Special handling for LoRA with sleep_level=2:
        # When sleep_level=2, base model weights are destroyed during each sleep cycle.
        # separately collect and update LoRA weights and base model weights through their respective interfaces.
        # Here: params contains LoRA weights, base_model_params contains base model weights.
        # Only needed if the rollout engine actually sleeps/frees weights (free_cache_engine=True).
        if (
            peft_config is not None
            and getattr(self.rollout, "sleep_level", None) == 2
            and self.config.rollout.free_cache_engine
        ):
            base_model_params = collect_lora_params(
                module=self.actor_module_fsdp,
                layered_summon=self.layered_summon,
                base_sync_done=False,
            )
            base_model_params = {replace_lora_wrapper(k, peft_config): v for k, v in base_model_params.items()}
            base_model_params = convert_weight_keys(
                base_model_params, getattr(self.actor_module_fsdp, "_fsdp_wrapped_module", self.actor_module_fsdp)
            )

        log_gpu_memory_usage("Before offload_fsdp_model_to_cpu", logger=logger)
        # breakpoint()
        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.actor_module_fsdp)
        log_gpu_memory_usage("After offload_fsdp_model_to_cpu", logger=logger)

        set_expandable_segments(False)

        # Ensure all params are on GPU before sending to rollout engine.
        # collect_lora_params returns CPU tensors via .detach().cpu(), so the
        # previous base_sync_done fast-path silently passed CPU tensors through,
        # which could cause errors in backends that expect GPU tensors (e.g.
        # SGLang's CUDA IPC serializer).
        device = get_device_id()
        _items = params.items() if isinstance(params, dict) else params
        per_tensor_param = (
            (
                name,
                param.to(device, non_blocking=True).full_tensor()
                if isinstance(param, DTensor)
                else param.to(device, non_blocking=False),
            )
            for name, param in _items
        )

        # QAT: quantize weights before sending to vLLM
        if self._qat_enabled:
            from verl.utils.qat.quantizer import QATQuantizer

            quantizer = QATQuantizer(
                mode=self.qat_config.mode,
                group_size=self.qat_config.group_size,
                ignore_patterns=self.qat_config.ignore_patterns,
                device=torch.device(get_device_id()),
                param_dtype=self._param_dtype,
            )
            per_tensor_param = quantizer.quantize_with_fusion(
                per_tensor_param,
                target_device=torch.device("cpu"),
            )
            aggressive_empty_cache(force_sync=True)

        if self.config.rollout.free_cache_engine:
            await self.rollout.resume(tags=["weights"])
        log_gpu_memory_usage("After resume weights", logger=logger)

        if (
            peft_config is not None
            and getattr(self.rollout, "sleep_level", None) == 2
            and self.config.rollout.free_cache_engine
        ):
            per_tensor_base_params = (
                (name, param.to(device, non_blocking=True).full_tensor() if isinstance(param, DTensor) else param)
                for name, param in base_model_params.items()
            )
            await self.rollout.update_weights(per_tensor_base_params, base_sync_done=False)
            del base_model_params, per_tensor_base_params

        await self.rollout.update_weights(per_tensor_param, peft_config=peft_config, base_sync_done=self.base_sync_done)
        log_gpu_memory_usage("After update_weights", logger=logger)
        del params, per_tensor_param
        aggressive_empty_cache(force_sync=True)
        if self.config.rollout.free_cache_engine:
            await self.rollout.resume(tags=["kv_cache"])
        log_gpu_memory_usage("After resume kv_cache", logger=logger)

        self.base_sync_done = True
        set_expandable_segments(True)

    def _build_model_optimizer(
        self,
        model_path,
        fsdp_config: FSDPEngineConfig,
        optim_config,
        override_model_config,
        use_remove_padding=False,
        use_fused_kernels=False,
        enable_gradient_checkpointing=False,
        trust_remote_code=False,
        use_liger=False,
        role="actor",
        enable_activation_offload=False,
        use_prefix_grouper=False,
        use_tiled_mlp=False,
        tiled_mlp_shards=4,
    ):
        from transformers import AutoConfig

        from verl.utils.model import get_generation_config, print_model_size, update_model_config
        from verl.utils.torch_dtypes import PrecisionType

        assert role in ["actor", "ref"]

        from typing import Any

        import yaml
        from mindspeed_mm.fsdp.params.argument import Arguments
        from mindspeed_mm.fsdp.params.utils import instantiate_dataclass

        mm_file_path = os.environ.get("MM_CONFIG_FILE")
        with open(os.path.abspath(mm_file_path), encoding="utf-8") as f:
            input_data: dict[str, dict[str, Any]] = yaml.safe_load(f)
        self.mm_args = instantiate_dataclass(Arguments, input_data)
        self.mm_args.training.compute_distributed_training(self.mm_args.parallel)

        # TiledMLP requires FSDP2 for correct gradient computation
        if use_tiled_mlp and self.config.actor.strategy == "fsdp":
            raise ValueError("TiledMLP requires FSDP2. Set `actor_rollout_ref.actor.strategy=fsdp2`.")

        log_gpu_memory_usage(f"Before init {role} from HF AutoModel", logger=logger)
        local_path = model_path

        # note that we have to create model in fp32. Otherwise, the optimizer is in bf16, which is incorrect
        self.tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
        self.processor = hf_processor(local_path, trust_remote_code=trust_remote_code)

        if self.config.model.get("custom_chat_template", None) is not None:
            if self.processor is not None:
                self.processor.chat_template = self.config.model.custom_chat_template
            else:
                self.tokenizer.chat_template = self.config.model.custom_chat_template

        torch_dtype = fsdp_config.get("model_dtype", None)
        if torch_dtype is None:
            torch_dtype = torch.float32 if self._is_actor else torch.bfloat16
        else:
            torch_dtype = PrecisionType.to_dtype(torch_dtype)

        # override model kwargs
        attn_implementation = override_model_config.get("attn_implementation", "flash_attention_2")
        actor_model_config = AutoConfig.from_pretrained(
            local_path, trust_remote_code=trust_remote_code, attn_implementation=attn_implementation
        )
        # which will be patched by _ulysses_flash_attention_forward, but errorly misses position_ids
        # Maybe support Ulysses in VisionAttention in the future and remove this patch
        if self.ulysses_sequence_parallel_size > 1 and hasattr(actor_model_config, "vision_config"):
            actor_model_config.vision_config._attn_implementation = "eager"

        # patch for qwen2.5-vl: when using flash_attention_3, set vision tower to use flash_attention_2
        # because the vision tower does not support flash_attention_3
        if (
            getattr(actor_model_config, "model_type", None) == "qwen2_5_vl"
            and attn_implementation == "flash_attention_3"
            and hasattr(actor_model_config, "vision_config")
        ):
            actor_model_config.vision_config._attn_implementation = "flash_attention_2"

        # patch for kimi-vl
        if getattr(actor_model_config, "model_type", None) == "kimi_vl":
            actor_model_config.text_config.topk_method = "greedy"

        self.generation_config = get_generation_config(local_path, trust_remote_code=trust_remote_code)

        override_config_kwargs = {
            "bos_token_id": self.tokenizer.bos_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "pad_token_id": self.tokenizer.pad_token_id,
        }

        if self.config.model.get("mtp", {}).get("enable", False):
            raise NotImplementedError("Right now,  MTP is not supported in FSDP")
        else:
            if hasattr(actor_model_config, "num_nextn_predict_layers"):
                actor_model_config.num_nextn_predict_layers = 0

        override_config_kwargs.update(override_model_config)
        update_model_config(actor_model_config, override_config_kwargs=override_config_kwargs)
        if self.rank == 0:
            print(f"Model config after override: {actor_model_config}")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from mindspeed_mm.fsdp.train.trainer import Trainer

            if role == "actor" and fsdp_config.offload_policy:
                self.mm_args.parallel.fsdp_plan.cpu_offload = True
                self._is_offload_param = False
                self._is_offload_optimizer = False

            if role == "ref":
                self.mm_args.parallel.fsdp_plan.cpu_offload = True

            trainer = Trainer(args=self.mm_args, dataloader_provider=self._dataloader)
            trainer.train_dataloader = None
            actor_module = trainer.model
            self.trainer = trainer

            # Apply Liger kernel to the model if use_liger is set to True
            if use_liger:
                from liger_kernel.transformers.monkey_patch import _apply_liger_kernel_to_instance

                _apply_liger_kernel_to_instance(model=actor_module)

            fused_kernel_options = self.config.model.get("fused_kernel_options", None)
            fused_kernels_backend = (
                fused_kernel_options.get("impl_backend", None) if fused_kernel_options is not None else None
            )

            apply_monkey_patch(
                model=actor_module,
                use_remove_padding=use_remove_padding,
                ulysses_sp_size=self.ulysses_sequence_parallel_size,
                use_fused_kernels=use_fused_kernels,
                fused_kernels_backend=fused_kernels_backend,
                use_prefix_grouper=use_prefix_grouper,
                use_tiled_mlp=use_tiled_mlp,
                tiled_mlp_shards=tiled_mlp_shards,
            )

            # some parameters may not in torch_dtype.
            actor_module.to(torch_dtype)

        self.use_orig_params = fsdp_config.get("use_orig_params", False)
        if self.config.actor.get("freeze_vision_tower", False):
            vision_tower = get_vl_model_vision_tower(actor_module)
            if vision_tower is not None:
                vision_tower.requires_grad_(False)
                self.use_orig_params = True
                if self.rank == 0:
                    print("[actor model] Vision tower is set to not trainable.")
            else:
                if self.rank == 0:
                    print("[actor model] No vision tower found.")

        torch.distributed.barrier()

        if self.rank == 0:
            print_model_size(actor_module)

        log_gpu_memory_usage(f"After init {role} from HF AutoModel", logger=logger)
        actor_module_fsdp = actor_module

        log_gpu_memory_usage(f"After {role} FSDP init", logger=logger)

        if role == "actor" and optim_config is not None:
            actor_optimizer = trainer.optimizer
            actor_lr_scheduler = trainer.lr_scheduler

            log_gpu_memory_usage(f"After {role} optimizer init", logger=logger)
        else:
            trainer.optimizer = None
            trainer.lr_scheduler = None
            actor_optimizer = None
            actor_lr_scheduler = None

        return actor_module_fsdp, actor_optimizer, actor_lr_scheduler, actor_model_config

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        super().init_model()
        from mindspeed_mm.fsdp.optimizer.clip_grad_norm import clip_grad_norm

        def _optimizer_step():
            grad_norm = clip_grad_norm(
                self.trainer.model,
                max_norm=self.mm_args.training.clip_grad,
                foreach=self.mm_args.training.clip_grad_foreach,
            )
            # Update parameters
            if self._is_actor:
                self.trainer.optimizer.step()
            return grad_norm

        if self._is_actor:
            self.actor._optimizer_step = _optimizer_step

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def save_checkpoint(self, local_path, hdfs_path=None, global_step=0, max_ckpt_to_keep=None):
        assert self._is_actor, "only support save checkpoint for actor"

        if self._is_offload_param:
            load_fsdp_model_to_gpu(self.actor_module_fsdp)

        state = {
            "model": self.actor_module_fsdp,
            "extra_state": {
                "iteration": global_step,
                "consumed_train_samples": 0,
                "lr_scheduler": self.actor_lr_scheduler.state_dict(),
            },
        }
        if not self.mm_args.training.no_save_optim:
            state["optimizer"] = self.actor_optimizer
        if not self.mm_args.training.no_save_rng:
            state["extra_state"]["torch_rng_state"] = torch.get_rng_state()

        self.trainer.checkpointer.save(
            path=local_path,
            state=state,
            iteration=global_step,
            save_async=self.mm_args.training.save_async,
        )

        if max_ckpt_to_keep is not None:
            self._cleanup_old_checkpoints(local_path, max_ckpt_to_keep)

        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.actor_module_fsdp)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def load_checkpoint(self, local_path, hdfs_path=None, del_local_after_load=False):
        assert self._is_actor or (not self._is_actor and self._is_rollout)

        if local_path is None:
            if self._is_offload_param:
                offload_fsdp_model_to_cpu(self.actor_module_fsdp)
            if self._is_offload_optimizer:
                offload_fsdp_optimizer(self.actor_optimizer)
            return

        if self._is_offload_param:
            load_fsdp_model_to_gpu(self.actor_module_fsdp)

        state = {"model": self.actor_module_fsdp, "extra_state": {}}
        if not self.mm_args.training.no_load_optim:
            state["optimizer"] = self.actor_optimizer

        release = self.trainer.checkpointer.load(
            path=local_path,
            state=state,
            load_rank0_and_broadcast=self.mm_args.training.load_rank0_and_broadcast,
            load_strict=self.mm_args.training.load_strict,
        )

        if not release and "extra_state" in state:
            if "lr_scheduler" in state["extra_state"]:
                self.actor_lr_scheduler.load_state_dict(state["extra_state"]["lr_scheduler"])
            if not self.mm_args.training.no_load_rng and "torch_rng_state" in state["extra_state"]:
                torch.set_rng_state(state["extra_state"]["torch_rng_state"])

        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.actor_module_fsdp)
        if self._is_offload_optimizer:
            offload_fsdp_optimizer(self.actor_optimizer)

    def _cleanup_old_checkpoints(self, base_path, max_to_keep):
        iter_pattern = re.compile(r"iter_(\d+)")
        iter_dirs = []
        for name in os.listdir(base_path):
            match = iter_pattern.match(name)
            if match:
                iter_dirs.append((int(match.group(1)), os.path.join(base_path, name)))

        iter_dirs.sort(key=lambda x: x[0])
        while len(iter_dirs) > max_to_keep:
            _, old_dir = iter_dirs.pop(0)
            if os.path.isdir(old_dir):
                shutil.rmtree(old_dir)


class AsyncMMActorRolloutRefWorker(AsyncActorRolloutRefWorker, MMActorRolloutRefWorker):
    pass
