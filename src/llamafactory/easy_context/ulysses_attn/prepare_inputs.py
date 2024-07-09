import torch


def extract_local(value, rank, world_size, device, dim=1):
    dimension_size = value.shape[dim]
    sub_seq_length = dimension_size // world_size

    sub_seq_start = rank * sub_seq_length
    sub_seq_end = (rank + 1) * sub_seq_length
    local_value = value[:, sub_seq_start:sub_seq_end]
    if device == None:
        return local_value
    return local_value.to(device)


def prepare_ulysses_attn_inputs(
    input_ids, position_ids, target_ids, rank, world_size, device
):

    local_input_ids = extract_local(
        input_ids,
        rank,
        world_size,
        device,
    )
    local_position_ids = extract_local(
        position_ids,
        rank,
        world_size,
        device,
    )

    if target_ids is not None:
        local_target_ids = extract_local(
            target_ids,
            rank,
            world_size,
            device,
        )
    else:
        local_target_ids = None
    return {
        "local_input_ids": local_input_ids,
        "local_position_ids": local_position_ids,
        "local_target_ids": local_target_ids,
    }

def prepare_ulysses_attn_sft_inputs(
    input_ids, attention_mask, position_ids, labels, rank, world_size, device
):
    local_input_ids = extract_local(
        input_ids,
        rank,
        world_size,
        device,
    )
    local_position_ids = extract_local(
        position_ids,
        rank,
        world_size,
        device,
    )
    local_attention_mask = extract_local(
        attention_mask,
        rank,
        world_size,
        device
    )
    local_labels = extract_local(
        labels,
        rank,
        world_size,
        device,
    )
    return {
        "input_ids": local_input_ids,
        "attention_mask": local_attention_mask,
        "position_ids": local_position_ids,
        "labels": local_labels,
    }