import torch
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set, Tuple

from llmtuner.extras.constants import LAYERNORM_NAMES
from llmtuner.extras.logging import get_logger

if TYPE_CHECKING:
    from transformers.modeling_utils import PreTrainedModel
    from llmtuner.hparams import ModelArguments, DataArguments, FinetuningArguments


logger = get_logger(__name__)


def find_all_linear_modules(
    model: "PreTrainedModel",
    quantization_bit: Optional[int] = None
) -> List[str]:
    if quantization_bit is not None:
        import bitsandbytes as bnb
        linear_cls = bnb.nn.Linear4bit if quantization_bit == 4 else bnb.nn.Linear8bitLt
    else:
        linear_cls = torch.nn.Linear

    output_layer_names = ["lm_head"]
    if model.config.model_type == "chatglm":
        output_layer_names.append("output_layer")

    module_names = set()
    for name, module in model.named_modules():
        if (
            isinstance(module, linear_cls)
            and not any([output_layer in name for output_layer in output_layer_names])
        ):
            module_names.add(name.split(".")[-1])

    logger.info("Found linear modules: {}".format(",".join(module_names)))
    return list(module_names)


def generate_model_card(
    model_args: "ModelArguments",
    data_args: "DataArguments",
    finetuning_args: "FinetuningArguments"
) -> Dict[str, Any]:
    return {
        "tasks": "text-generation",
        "finetuned_from": model_args.model_name_or_path,
        "dataset": [dataset.strip() for dataset in data_args.dataset.split(",")],
        "tags": ["llama-factory"] + (["lora"] if finetuning_args.finetuning_type == "lora" else [])
    }


def prepare_model_for_training(
    model: "PreTrainedModel",
    finetuning_args: "FinetuningArguments",
    output_layer_name: Optional[str] = "lm_head",
    use_gradient_checkpointing: Optional[bool] = True,
    layernorm_names: Optional[Set[str]] = LAYERNORM_NAMES
) -> "PreTrainedModel":
    r"""
    Includes:
        (1) cast the layernorm in fp32
        (2) make output embedding layer require grads
        (3) upcast the lm_head to fp32
    Inspired by: https://github.com/huggingface/peft/blob/v0.2.0/src/peft/utils/other.py#L33
    """
    if finetuning_args.upcast_layernorm:
        for name, param in model.named_parameters():
            if param.ndim == 1 and any(ln_name in name for ln_name in layernorm_names):
                param.data = param.data.to(torch.float32)
        logger.info("Upcasting weights in layernorm in float32.")

    if finetuning_args.neft_alpha > 1e-6:
        def neftune_forward_hook(module: torch.nn.Module, args: Tuple[torch.Tensor], output: torch.Tensor):
            if module.training:
                dims = torch.tensor(output.size(1) * output.size(2))
                mag_norm = finetuning_args.neft_alpha / torch.sqrt(dims)
                output = output + torch.zeros_like(output).uniform_(-mag_norm, mag_norm)
            return output

        model.get_input_embeddings().register_forward_hook(neftune_forward_hook)
        logger.info("Using noisy embedding with alpha={:.2f}".format(finetuning_args.neft_alpha))

    if use_gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module: torch.nn.Module, args: Tuple[torch.Tensor], output: torch.Tensor):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

        model.gradient_checkpointing_enable()
        model.config.use_cache = False # turn off when gradient checkpointing is enabled
        logger.info("Gradient checkpointing enabled.")

    if finetuning_args.finetuning_type != "full" and hasattr(model, output_layer_name):
        output_layer = getattr(model, output_layer_name)
        if isinstance(output_layer, torch.nn.Linear):
            def fp32_forward_pre_hook(module: torch.nn.Module, args: Tuple[torch.Tensor]):
                return args[0].to(output_layer.weight.dtype)
            def fp32_forward_post_hook(module: torch.nn.Module, args: Tuple[torch.Tensor], output: torch.Tensor):
                return output.to(torch.float32)
            output_layer.register_forward_pre_hook(fp32_forward_pre_hook)
            output_layer.register_forward_hook(fp32_forward_post_hook)

    return model
