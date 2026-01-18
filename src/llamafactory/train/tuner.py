# Copyright 2025 the KVCache.AI team, Approaching AI, and the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import shutil
from typing import TYPE_CHECKING, Any, Optional

import torch
import torch.distributed as dist
from transformers import EarlyStoppingCallback, PreTrainedModel
from transformers.utils import is_torch_cuda_available, is_torch_npu_available

from ..data import get_template_and_fix_tokenizer
from ..extras import logging
from ..extras.constants import V_HEAD_SAFE_WEIGHTS_NAME, V_HEAD_WEIGHTS_NAME
from ..extras.misc import infer_optim_dtype
from ..extras.packages import is_mcore_adapter_available, is_ray_available
from ..hparams import get_infer_args, get_ray_args, get_train_args, read_args
from ..model import load_model, load_tokenizer
from .callbacks import LogCallback, PissaConvertCallback, ReporterCallback
from .dpo import run_dpo
from .kto import run_kto
from .ppo import run_ppo
from .pt import run_pt
from .rm import run_rm
from .sft import run_sft
from .trainer_utils import (
    get_master_addr_port,
    get_placement_group,
    get_ray_remote_config_for_worker,
    get_swanlab_callback,
)


if is_ray_available():
    import ray
    from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy


if TYPE_CHECKING:
    from transformers import TrainerCallback


logger = logging.get_logger(__name__)


def _training_function(config: dict[str, Any]) -> None:
    args = config.get("args")
    callbacks: list[Any] = config.get("callbacks")
    model_args, data_args, training_args, finetuning_args, generating_args = get_train_args(args)

    callbacks.append(LogCallback())
    if finetuning_args.pissa_convert:
        callbacks.append(PissaConvertCallback())

    if finetuning_args.use_swanlab:
        callbacks.append(get_swanlab_callback(finetuning_args))

    if finetuning_args.early_stopping_steps is not None:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=finetuning_args.early_stopping_steps))

    callbacks.append(ReporterCallback(model_args, data_args, finetuning_args, generating_args))  # add to last

    if finetuning_args.stage in ["pt", "sft", "dpo"] and finetuning_args.use_mca:
        if not is_mcore_adapter_available():
            raise ImportError("mcore_adapter is not installed. Please install it with `pip install mcore-adapter`.")
        if finetuning_args.stage == "pt":
            from .mca import run_pt as run_pt_mca

            run_pt_mca(model_args, data_args, training_args, finetuning_args, callbacks)
        elif finetuning_args.stage == "sft":
            from .mca import run_sft as run_sft_mca

            run_sft_mca(model_args, data_args, training_args, finetuning_args, callbacks)
        elif finetuning_args.stage == "dpo":
            from .mca import run_dpo as run_dpo_mca

            run_dpo_mca(model_args, data_args, training_args, finetuning_args, callbacks)

    elif finetuning_args.stage == "pt":
        run_pt(model_args, data_args, training_args, finetuning_args, callbacks)
    elif finetuning_args.stage == "sft":
        run_sft(model_args, data_args, training_args, finetuning_args, generating_args, callbacks)
    elif finetuning_args.stage == "rm":
        run_rm(model_args, data_args, training_args, finetuning_args, callbacks)
    elif finetuning_args.stage == "ppo":
        run_ppo(model_args, data_args, training_args, finetuning_args, generating_args, callbacks)
    elif finetuning_args.stage == "dpo":
        run_dpo(model_args, data_args, training_args, finetuning_args, callbacks)
    elif finetuning_args.stage == "kto":
        run_kto(model_args, data_args, training_args, finetuning_args, callbacks)
    else:
        raise ValueError(f"Unknown task: {finetuning_args.stage}.")

    if is_ray_available() and ray.is_initialized():
        return  # if ray is intialized it will destroy the process group on return

    try:
        if dist.is_initialized():
            dist.destroy_process_group()
    except Exception as e:
        logger.warning(f"Failed to destroy process group: {e}.")


def run_exp(args: Optional[dict[str, Any]] = None, callbacks: Optional[list["TrainerCallback"]] = None) -> None:
    args = read_args(args)
    if "-h" in args or "--help" in args:
        get_train_args(args)

    ray_args = get_ray_args(args)
    callbacks = callbacks or []
    if ray_args.use_ray:
        _ray_training_function(ray_args, args, callbacks)
    else:
        _training_function(config={"args": args, "callbacks": callbacks})


def export_model(args: Optional[dict[str, Any]] = None) -> None:
    model_args, data_args, finetuning_args, _ = get_infer_args(args)

    if model_args.export_dir is None:
        raise ValueError("Please specify `export_dir` to save model.")

    if model_args.adapter_name_or_path is not None and model_args.export_quantization_bit is not None:
        raise ValueError("Please merge adapters before quantizing the model.")

    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]
    processor = tokenizer_module["processor"]
    template = get_template_and_fix_tokenizer(tokenizer, data_args)
    model = load_model(tokenizer, model_args, finetuning_args)  # must after fixing tokenizer to resize vocab

    if getattr(model, "quantization_method", None) is not None and model_args.adapter_name_or_path is not None:
        raise ValueError("Cannot merge adapters to a quantized model.")

    if not isinstance(model, PreTrainedModel):
        raise ValueError("The model is not a `PreTrainedModel`, export aborted.")

    if getattr(model, "quantization_method", None) is not None:  # quantized model adopts float16 type
        setattr(model.config, "torch_dtype", torch.float16)
    else:
        if model_args.infer_dtype == "auto":
            output_dtype = getattr(model.config, "torch_dtype", torch.float32)
            if output_dtype == torch.float32:  # if infer_dtype is auto, try using half precision first
                output_dtype = infer_optim_dtype(torch.bfloat16)
        else:
            output_dtype = getattr(torch, model_args.infer_dtype)

        setattr(model.config, "torch_dtype", output_dtype)
        model = model.to(output_dtype)
        logger.info_rank0(f"Convert model dtype to: {output_dtype}.")

    model.save_pretrained(
        save_directory=model_args.export_dir,
        max_shard_size=f"{model_args.export_size}GB",
        safe_serialization=(not model_args.export_legacy_format),
    )
    if model_args.export_hub_model_id is not None:
        model.push_to_hub(
            model_args.export_hub_model_id,
            token=model_args.hf_hub_token,
            max_shard_size=f"{model_args.export_size}GB",
            safe_serialization=(not model_args.export_legacy_format),
        )

    if finetuning_args.stage == "rm":
        if model_args.adapter_name_or_path is not None:
            vhead_path = model_args.adapter_name_or_path[-1]
        else:
            vhead_path = model_args.model_name_or_path

        if os.path.exists(os.path.join(vhead_path, V_HEAD_SAFE_WEIGHTS_NAME)):
            shutil.copy(
                os.path.join(vhead_path, V_HEAD_SAFE_WEIGHTS_NAME),
                os.path.join(model_args.export_dir, V_HEAD_SAFE_WEIGHTS_NAME),
            )
            logger.info_rank0(f"Copied valuehead to {model_args.export_dir}.")
        elif os.path.exists(os.path.join(vhead_path, V_HEAD_WEIGHTS_NAME)):
            shutil.copy(
                os.path.join(vhead_path, V_HEAD_WEIGHTS_NAME),
                os.path.join(model_args.export_dir, V_HEAD_WEIGHTS_NAME),
            )
            logger.info_rank0(f"Copied valuehead to {model_args.export_dir}.")

    try:
        tokenizer.padding_side = "left"  # restore padding side
        tokenizer.init_kwargs["padding_side"] = "left"
        tokenizer.save_pretrained(model_args.export_dir)
        if model_args.export_hub_model_id is not None:
            tokenizer.push_to_hub(model_args.export_hub_model_id, token=model_args.hf_hub_token)

        if processor is not None:
            processor.save_pretrained(model_args.export_dir)
            if model_args.export_hub_model_id is not None:
                processor.push_to_hub(model_args.export_hub_model_id, token=model_args.hf_hub_token)

    except Exception as e:
        logger.warning_rank0(f"Cannot save tokenizer, please copy the files manually: {e}.")

    ollama_modelfile = os.path.join(model_args.export_dir, "Modelfile")
    with open(ollama_modelfile, "w", encoding="utf-8") as f:
        f.write(template.get_ollama_modelfile(tokenizer))
        logger.info_rank0(f"Ollama modelfile saved in {ollama_modelfile}")


@ray.remote
class RayWorker:
    def __init__(self):
        self._setup_env_visible_devices()

        rank = os.environ["RANK"]
        world_size = os.environ["WORLD_SIZE"]
        master_addr = os.environ["MASTER_ADDR"]
        master_port = os.environ["MASTER_PORT"]
        use_agent_store = os.environ["TORCHELASTIC_USE_AGENT_STORE"]
        local_rank = os.environ["LOCAL_RANK"] if os.environ.get("LOCAL_RANK", None) else "0"

        store = {
            "RANK": rank,
            "WORLD_SIZE": world_size,
            "MASTER_ADDR": master_addr,
            "MASTER_PORT": master_port,
            "LOCAL_RANK": local_rank,
            "TORCHELASTIC_USE_AGENT_STORE": use_agent_store,
        }

        visible_devices_keyword = "ASCEND_RT_VISIBLE_DEVICES" if is_torch_npu_available() else "CUDA_VISIBLE_DEVICES"
        visible_devices_id = os.environ.get(visible_devices_keyword)
        store[visible_devices_keyword] = visible_devices_id

        for key, value in store.items():
            if value is not None:
                os.environ[key] = value

        if is_torch_cuda_available():
            torch.cuda.set_device(int(local_rank))
        elif is_torch_npu_available():
            torch.npu.set_device(int(local_rank))

    def _setup_env_visible_devices(self):
        RAY_NOSET_VISIBLE_DEVICES_LIST = [
            "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES",
            "RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES",
        ]
        is_ray_noset_visible_devices = any(os.environ.get(env_var, None) for env_var in RAY_NOSET_VISIBLE_DEVICES_LIST)
        if is_ray_noset_visible_devices:
            device_name = "NPU" if is_torch_npu_available() else "GPU"
            local_rank = ray.get_runtime_context().get_accelerator_ids()[device_name][0]
            os.environ["LOCAL_RANK"] = local_rank

    def _training_function(self, config):
        _training_function(config)


def _ray_training_function(ray_args, args, callbacks):
    num_workers = ray_args.ray_num_workers
    logger.info(f"Using ray.remote mode with {num_workers} workers for distributed training.")

    # initialize ray
    if not ray.is_initialized():
        if ray_args.ray_init_kwargs is not None:
            ray.init(**ray_args.ray_init_kwargs)
        else:
            ray.init()

    # create placementgroup for resource management
    pg, bundle = get_placement_group(num_workers, ray_args.placement_strategy)
    ray.get(pg.ready())
    logger.info(f"Create placement group with {num_workers} bundles: {bundle}")

    # get master_addr_port
    master_addr, master_port = ray.get(
        get_master_addr_port.options(
            scheduling_strategy=PlacementGroupSchedulingStrategy(
                placement_group=pg,
                placement_group_bundle_index=0,
            ),
        ).remote()
    )

    # luanch workers
    workers = []
    for rank in range(num_workers):
        remote_config = get_ray_remote_config_for_worker(
            placement_group=pg,
            bundle_idx=rank,
            rank=rank,
            world_size=num_workers,
            master_addr=master_addr,
            master_port=master_port,
        )
        worker = RayWorker.options(**remote_config).remote()
        workers.append(worker)

    ray.get([worker._training_function.remote(config={"args": args, "callbacks": callbacks}) for worker in workers])
    ray.shutdown()
