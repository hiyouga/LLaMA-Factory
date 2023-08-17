import torch
from typing import TYPE_CHECKING, List, Optional, Tuple
from transformers import InfNanRemoveLogitsProcessor, LogitsProcessorList

from llmtuner.extras.constants import LAYERNORM_NAMES

if TYPE_CHECKING:
    from transformers.modeling_utils import PreTrainedModel


class AverageMeter:
    r"""
    Computes and stores the average and current value.
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def get_logits_processor() -> LogitsProcessorList:
    logits_processor = LogitsProcessorList()
    logits_processor.append(InfNanRemoveLogitsProcessor())
    return logits_processor


def count_parameters(model: torch.nn.Module) -> Tuple[int, int]:
    r"""
    Returns the number of trainable parameters and number of all parameters in the model.
    """
    trainable_params, all_param = 0, 0
    for param in model.parameters():
        num_params = param.numel()
        # if using DS Zero 3 and the weights are initialized empty
        if num_params == 0 and hasattr(param, "ds_numel"):
            num_params = param.ds_numel

        # Due to the design of 4bit linear layers from bitsandbytes, multiply the number of parameters by 2
        if param.__class__.__name__ == "Params4bit":
            num_params = num_params * 2

        all_param += num_params
        if param.requires_grad:
            trainable_params += num_params

    return trainable_params, all_param


# Includes: (1) cast the layernorm in fp32 (2) make output embedding layer require grads (3) upcast the lm_head to fp32
# Inspired by: https://github.com/huggingface/peft/blob/c0209c35abbf88c63aa267800d98a8e212ed0a42/src/peft/utils/other.py#L35
def prepare_model_for_training(
    model: "PreTrainedModel",
    finetuning_type: str,
    output_layer_name: Optional[str] = "lm_head",
    use_gradient_checkpointing: Optional[bool] = True,
    layer_norm_names: Optional[List[str]] = LAYERNORM_NAMES
) -> "PreTrainedModel":
    for name, param in model.named_parameters():
        if param.ndim == 1 and any(layer_norm_name in name for layer_norm_name in layer_norm_names):
            param.data = param.data.to(torch.float32)

    if use_gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

        model.gradient_checkpointing_enable()
        model.config.use_cache = False # turn off when gradient checkpointing is enabled

    if finetuning_type != "full" and hasattr(model, output_layer_name):
        output_layer: torch.nn.Linear = getattr(model, output_layer_name)
        input_dtype = output_layer.weight.dtype

        class CastOutputToFloat(torch.nn.Sequential):

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return super().forward(x.to(input_dtype)).to(torch.float32)

        setattr(model, output_layer_name, CastOutputToFloat(output_layer))

    return model


def torch_gc() -> None:
    r"""
    Collects GPU memory.
    """
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


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
