from .dist_flash_attn.prepare_input import prepare_dist_flash_attn_inputs, prepare_dist_flash_attn_sft_inputs
from .dist_flash_attn.monkey_patch import apply_dist_flash_attn_monkey_patch_llama
from .zigzag_ring_attn.prepare_inputs import prepare_zigzag_ring_attn_inputs, prepare_zigzag_ring_attn_sft_inputs
from .zigzag_ring_attn.monkey_patch import apply_zigzag_ring_attn_monkey_patch_llama    
from .zigzag_ring_attn.monkey_patch import apply_zigzag_ring_attn_monkey_patch_mistral
from .unsloth_offloaded_gradient_checkpoint.monkey_patch import apply_unsloth_offloaded_gradient_checkpoint_monkey_patch
from .ulysses_attn.prepare_inputs import prepare_ulysses_attn_inputs, prepare_ulysses_attn_sft_inputs
from .ulysses_attn.monkey_patch import apply_ulysses_attn_monkey_patch_llama 
import torch
import torch.nn.functional as F

def prepare_seq_parallel_inputs(
    seq_algo, input_ids, position_ids, target_ids, rank, world_size, device
):
    if seq_algo == "zigzag_ring_attn":
        return prepare_zigzag_ring_attn_inputs(
            input_ids, position_ids, target_ids, rank, world_size, device
        )
    elif seq_algo == "dist_flash_attn":
        return prepare_dist_flash_attn_inputs(
            input_ids, position_ids, target_ids, rank, world_size, device
        )
    elif seq_algo == "ulysses_attn":
        return prepare_ulysses_attn_inputs(
            input_ids, position_ids, target_ids, rank, world_size, device
        )
    elif seq_algo == "data_parallel":
        return {
            "local_input_ids": input_ids.to(device),
            "local_position_ids": position_ids.to(device),
            "local_target_ids": target_ids.to(device),
        }
    else:
        raise ValueError(f"Invalid seq_algo: {seq_algo}")
    
def prepare_seq_parallel_sft_inputs(
    seq_algo, input_ids, attention_mask, position_ids, labels, rank, world_size, device
):
    if attention_mask is not None and position_ids is None:
        # create position_ids on the fly for batch generation
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)
    shift_labels = F.pad(labels, [0, 1], 'constant', -100)[:, 1:]
    if seq_algo == "zigzag_ring_attn":
        return prepare_zigzag_ring_attn_sft_inputs(
            input_ids, attention_mask, position_ids, shift_labels, rank, world_size, device
        )
    elif seq_algo == "dist_flash_attn":
        return prepare_dist_flash_attn_sft_inputs(
            input_ids, attention_mask, position_ids, shift_labels, rank, world_size, device
        )
    elif seq_algo == "ulysses_attn":
        return prepare_ulysses_attn_sft_inputs(
            input_ids, attention_mask, position_ids, shift_labels, rank, world_size, device
        )
    elif seq_algo == "data_parallel":
        return {
            "input_ids": input_ids,
            "position_ids": position_ids,
            "attention_mask": attention_mask,
            "target_ids": labels,
        }
    else:
        raise ValueError(f"Invalid seq_algo: {seq_algo}")
    
def apply_seq_parallel_monkey_patch(
    seq_algo, model,seq_parallel_size=None
):
    assert seq_algo in ["zigzag_ring_attn", "dist_flash_attn", "ulysses_attn", "data_parallel"], f"Invalid seq_algo: {seq_algo}"
    assert model in ["llama", "mistral"], f"Invalid model: {model}"
    if seq_algo == "data_parallel":
        return
    elif seq_algo == "zigzag_ring_attn" and model == "llama":
        apply_zigzag_ring_attn_monkey_patch_llama()
    elif seq_algo == "zigzag_ring_attn" and model == "mistral":
        apply_zigzag_ring_attn_monkey_patch_mistral()
    elif seq_algo == "dist_flash_attn" and model == "llama":
        apply_dist_flash_attn_monkey_patch_llama(seq_parallel_size=seq_parallel_size)
    elif seq_algo == "ulysses_attn" and model == "llama":
        apply_ulysses_attn_monkey_patch_llama()
    else:
        raise ValueError(f"Invalid seq_algo: {seq_algo} or model: {model}")
        
def prepare_dataloader(seq_algo, dataloader, acclerator):
    if seq_algo == "data_parallel":
        return acclerator.prepare(dataloader)
    else:
        return dataloader
