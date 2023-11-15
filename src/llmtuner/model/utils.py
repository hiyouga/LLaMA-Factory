import torch
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set, Tuple

from transformers.utils import cached_file
from transformers.trainer import WEIGHTS_NAME, SAFE_WEIGHTS_NAME

from llmtuner.extras.constants import LAYERNORM_NAMES
from llmtuner.extras.logging import get_logger
from llmtuner.hparams import ModelArguments, FinetuningArguments

if TYPE_CHECKING:
    from transformers.modeling_utils import PreTrainedModel
    from llmtuner.hparams import DataArguments


logger = get_logger(__name__)


def dispatch_model(model: "PreTrainedModel") -> "PreTrainedModel":
    r"""
    Dispatches a pre-trained model to GPUs with balanced memory.
    Borrowed from: https://github.com/huggingface/transformers/blob/v4.31.0/src/transformers/modeling_utils.py#L2803
    """
    if getattr(model, "is_loaded_in_8bit", False) or getattr(model, "is_loaded_in_4bit", False): # do nothing
        return model

    if torch.cuda.device_count() > 1:
        from accelerate import dispatch_model
        from accelerate.utils import infer_auto_device_map, get_balanced_memory

        if model._no_split_modules is None:
            raise ValueError("The model class needs to implement the `_no_split_modules` attribute.")

        kwargs = {"dtype": model.dtype, "no_split_module_classes": model._no_split_modules}
        max_memory = get_balanced_memory(model, **kwargs)
        # Make sure tied weights are tied before creating the device map.
        model.tie_weights()
        device_map = infer_auto_device_map(model, max_memory=max_memory, **kwargs)
        return dispatch_model(model, device_map)
    else:
        return model.cuda()


def find_all_linear_modules(
    model: "PreTrainedModel",
    quantization_bit: Optional[int] = None
) -> List[str]:
    r"""
    Finds all available modules to apply lora.
    """
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


def load_valuehead_params(
    path_or_repo_id: str,
    model_args: "ModelArguments"
) -> Dict[str, torch.Tensor]:
    r"""
    Loads value head parameters from Hugging Face Hub or local disk.

    Returns: dict with keys `v_head.summary.weight` and `v_head.summary.bias`.
    """
    kwargs = {
        "path_or_repo_id": path_or_repo_id,
        "cache_dir": model_args.cache_dir,
        "token": model_args.hf_hub_token
    }
    try:
        vhead_file = cached_file(filename=WEIGHTS_NAME, **kwargs)
    except:
        try:
            vhead_file = cached_file(filename=SAFE_WEIGHTS_NAME, **kwargs)
        except:
            logger.warning("Provided path ({}) does not contain valuehead weights.".format(path_or_repo_id))
            return None

    return torch.load(vhead_file, map_location="cpu")


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
