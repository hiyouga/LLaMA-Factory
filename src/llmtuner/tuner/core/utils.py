import torch
from types import MethodType
from typing import TYPE_CHECKING, List, Optional

from llmtuner.extras.constants import LAYERNORM_NAMES

if TYPE_CHECKING:
    from transformers.modeling_utils import PreTrainedModel
    from llmtuner.hparams import FinetuningArguments


def find_all_linear_modules(
    model: "PreTrainedModel",
    quantization_bit: Optional[int] = None,
    output_layer_name: Optional[str] = "lm_head"
) -> List[str]:
    if quantization_bit is not None:
        import bitsandbytes as bnb
        linear_cls = bnb.nn.Linear4bit if quantization_bit == 4 else bnb.nn.Linear8bitLt
    else:
        linear_cls = torch.nn.Linear

    module_names = set()
    for name, module in model.named_modules():
        if output_layer_name not in name and isinstance(module, linear_cls):
            module_names.add(name.split(".")[-1])

    if output_layer_name in module_names:
        module_names.pop(output_layer_name)

    return list(module_names)


def prepare_model_for_training(
    model: "PreTrainedModel",
    finetuning_args: "FinetuningArguments",
    output_layer_name: Optional[str] = "lm_head",
    use_gradient_checkpointing: Optional[bool] = True,
    layernorm_names: Optional[List[str]] = LAYERNORM_NAMES
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

    if finetuning_args.neft_alpha > 1e-6:
        input_embed: torch.nn.Embedding = model.get_input_embeddings()
        
        def noisy_forward(self: torch.nn.Embedding, x: torch.Tensor) -> torch.Tensor:
            embeddings = input_embed.forward(x)
            if self.training:
                dims = self.num_embeddings * self.embedding_dim
                mag_norm = finetuning_args.neft_alpha / (dims ** 0.5)
                embeddings += torch.zeros_like(embeddings).uniform_(-mag_norm, mag_norm)
            return embeddings

        input_embed.forward = MethodType(noisy_forward, input_embed)

    if use_gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module: torch.nn.Module, input: torch.Tensor, output: torch.Tensor):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

        model.gradient_checkpointing_enable()
        model.config.use_cache = False # turn off when gradient checkpointing is enabled

    if finetuning_args.finetuning_type != "full" and hasattr(model, output_layer_name):
        output_layer: torch.nn.Linear = getattr(model, output_layer_name)
        input_dtype = output_layer.weight.dtype

        def forward_in_fp32(self, x: torch.Tensor) -> torch.Tensor:
            return output_layer.forward(x.to(input_dtype)).to(torch.float32)

        output_layer.forward = MethodType(forward_in_fp32, output_layer)

    return model
