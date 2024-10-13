# Copyright 2024 the LlamaFactory team.
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
import gc
import os
import shutil
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import torch
from transformers import PreTrainedModel, AutoModelForVision2Seq, AutoModelForCausalLM
from transformers.utils.versions import require_version

from ..data import get_template_and_fix_tokenizer
from ..extras.constants import V_HEAD_SAFE_WEIGHTS_NAME, V_HEAD_WEIGHTS_NAME
from ..extras.logging import get_logger
from ..hparams import get_infer_args, get_train_args
from ..hparams.parser import _parse_train_args
from ..model import load_model, load_tokenizer
from .callbacks import LogCallback
from .dpo import run_dpo
from .kto import run_kto
from .ppo import run_ppo
from .pt import run_pt
from .rm import run_rm
from .sft import run_sft
from ..model.loader import _get_init_kwargs, load_config
from ..model.patcher import patch_config

if TYPE_CHECKING:
    from transformers import TrainerCallback

logger = get_logger(__name__)


def run_exp(args: Optional[Dict[str, Any]] = None, callbacks: List["TrainerCallback"] = []) -> None:
    callbacks.append(LogCallback())
    model_args, data_args, training_args, finetuning_args, generating_args = get_train_args(args)

    if finetuning_args.stage == "pt":
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
        raise ValueError("Unknown task: {}.".format(finetuning_args.stage))


def quantize_model(args: Optional[Dict[str, Any]] = None):
    if args is not None:
        res = args
    import sys
    if len(sys.argv) == 2 and sys.argv[1].endswith(".yaml"):
        yaml_file = os.path.abspath(sys.argv[1])
        import yaml
        from pathlib import Path
        res = yaml.safe_load(Path(yaml_file).read_text())

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        json_file = os.path.abspath(sys.argv[1])
        import json
        with open(Path(json_file), encoding="utf-8") as open_json_file:
            res = json.loads(open_json_file.read())
    model_args, data_args, training_args, finetuning_args, generating_args = _parse_train_args(args)
    if not (model_args.export_quantization_bit is not None and model_args.export_quantization_method == "auto_round"):
        return res

    if model_args.export_quantization_bit not in [4]:
        raise ValueError("AutoRound only accepts 4 bits quantization.")

    require_version("auto_round>=0.3.0", "To fix: pip install auto_round>=0.3.0")
    require_version("auto_gptq>=0.7.1", "To fix: pip install auto_gptq>=0.7.1")

    if model_args.adapter_name_or_path:
        raise ValueError("Please merge adapters before quantizing the model.")

    if model_args.mixture_of_depths == "load":
        raise NotImplementedError("AutoRound only supports `AutoModelForCausalLM` models ")
    if model_args.train_from_scratch:
        raise NotImplementedError("AutoRound only supports trained models")

    if model_args.export_dir is None:
        export_dir = 'saves/autoround_quantized_model'
        logger.warning(" `export_dir` has not been specified, set it to `saves/autoround_quantized_model`.")
    else:
        export_dir = model_args.export_dir

    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]
    # get_template_and_fix_tokenizer(tokenizer, data_args)
    from transformers.integrations import is_deepspeed_zero3_enabled
    from transformers.modeling_utils import is_fsdp_enabled
    if is_deepspeed_zero3_enabled() or is_fsdp_enabled():
        raise ValueError("DeepSpeed ZeRO-3 or FSDP is incompatible with PTQ-quantized models.")

    init_kwargs = _get_init_kwargs(model_args)
    config = load_config(model_args)
    if type(config) in AutoModelForVision2Seq._model_mapping.keys():
        raise NotImplementedError("AutoRound only supports `AutoModelForCausalLM` models ")
    patch_config(config, tokenizer, model_args, init_kwargs, is_trainable=False)
    init_kwargs["config"] = config
    init_kwargs["pretrained_model_name_or_path"] = model_args.model_name_or_path
    init_kwargs["device_map"] = "cpu"
    init_kwargs["torch_dtype"] = "auto"
    init_kwargs['config'].use_cache = False
    model = AutoModelForCausalLM.from_pretrained(**init_kwargs)
    bits, group_size, sym = model_args.export_quantization_bit, 128, False
    from auto_round import AutoRound
    autoround = AutoRound(model, tokenizer, bits=model_args.export_quantization_bit, group_size=group_size, sym=sym,
                          nsamples=2, iters=2)  ##TODO pass more configs and change it back
    autoround.quantize()
    autoround.save_quantized(export_dir, format='auto_gptq', inplace=True)
    torch.cuda.empty_cache()
    gc.collect()
    res.pop("export_quantization_bit")
    res["model_name_or_path"] = export_dir

    return res


def export_model(args: Optional[Dict[str, Any]] = None) -> None:
    model_args, data_args, finetuning_args, _ = get_infer_args(args)

    if model_args.export_dir is None:
        raise ValueError("Please specify `export_dir` to save model.")

    if model_args.adapter_name_or_path is not None and model_args.export_quantization_bit is not None:
        raise ValueError("Please merge adapters before quantizing the model.")

    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]
    processor = tokenizer_module["processor"]
    get_template_and_fix_tokenizer(tokenizer, data_args)
    model = load_model(tokenizer, model_args, finetuning_args)  # must after fixing tokenizer to resize vocab

    if getattr(model, "quantization_method", None) is not None and model_args.adapter_name_or_path is not None:
        raise ValueError("Cannot merge adapters to a quantized model.")

    if not isinstance(model, PreTrainedModel):
        raise ValueError("The model is not a `PreTrainedModel`, export aborted.")

    if getattr(model, "quantization_method", None) is not None:  # quantized model adopts float16 type
        setattr(model.config, "torch_dtype", torch.float16)
    else:
        if model_args.infer_dtype == "auto":
            output_dtype = getattr(model.config, "torch_dtype", torch.float16)
        else:
            output_dtype = getattr(torch, model_args.infer_dtype)

        setattr(model.config, "torch_dtype", output_dtype)
        model = model.to(output_dtype)
        logger.info("Convert model dtype to: {}.".format(output_dtype))

    model.save_pretrained(
        save_directory=model_args.export_dir,
        max_shard_size="{}GB".format(model_args.export_size),
        safe_serialization=(not model_args.export_legacy_format),
    )
    if model_args.export_hub_model_id is not None:
        model.push_to_hub(
            model_args.export_hub_model_id,
            token=model_args.hf_hub_token,
            max_shard_size="{}GB".format(model_args.export_size),
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
            logger.info("Copied valuehead to {}.".format(model_args.export_dir))
        elif os.path.exists(os.path.join(vhead_path, V_HEAD_WEIGHTS_NAME)):
            shutil.copy(
                os.path.join(vhead_path, V_HEAD_WEIGHTS_NAME),
                os.path.join(model_args.export_dir, V_HEAD_WEIGHTS_NAME),
            )
            logger.info("Copied valuehead to {}.".format(model_args.export_dir))

    try:
        tokenizer.padding_side = "left"  # restore padding side
        tokenizer.init_kwargs["padding_side"] = "left"
        tokenizer.save_pretrained(model_args.export_dir)
        if model_args.export_hub_model_id is not None:
            tokenizer.push_to_hub(model_args.export_hub_model_id, token=model_args.hf_hub_token)

        if processor is not None:
            getattr(processor, "image_processor").save_pretrained(model_args.export_dir)
            if model_args.export_hub_model_id is not None:
                getattr(processor, "image_processor").push_to_hub(
                    model_args.export_hub_model_id, token=model_args.hf_hub_token
                )

    except Exception as e:
        logger.warning("Cannot save tokenizer, please copy the files manually: {}.".format(e))
