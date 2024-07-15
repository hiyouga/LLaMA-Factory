"""
Materialization-aware gradient checkpointing monkey patch.
"""
from typing import List, Optional, Tuple

import torch
from torch import nn
from torch.utils.checkpoint import _get_autocast_kwargs, check_backward_validity, get_device_states, set_device_states, detach_variable

import transformers
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, BaseModelOutputWithPast

from einops import rearrange

from .lightseq_async_attn import _lightseq_forward, _lightseq_backward
from .async_communication import initialize_distributed, reset_global_memory_buffer

# define a global buffer to save flash attention outputs
# it's called global because it saves the outputs for all layers
global_flash_attn_out_buffer = None

# define a local buffer to save recomputed qkv
# it's called local because it's a temporary buffer which will be updated across layers
local_res_grad_buffer = None

# hooks for the gradients of residual
global_hooks = []

def init_flash_attn_buffers(num_layers):
    # update the global buffer according to number of layers
    global global_flash_attn_out_buffer
    global_flash_attn_out_buffer = [None] * num_layers
    
def clean_hook():
    # Remove all hooks in the global buffer
    for hook in global_hooks:
        hook.remove()
    # Clear the global buffer
    global_hooks.clear()

def clear_all_buffers_at_the_end_of_training():
    # call it at the end of training 
    global lobal_flash_attn_out_buffer
    global_flash_attn_out_buffer = None
    global local_res_grad_buffer
    local_res_grad_buffer = None
    clean_hook()

def save_flash_attn_out_to_global_buffer(idx, out):
    global global_flash_attn_out_buffer
    global_flash_attn_out_buffer[idx] = out

def get_flash_attn_out_from_global_buffer(idx):
    global global_flash_attn_out_buffer
    return global_flash_attn_out_buffer[idx]

def free_flash_attn_out_buffer(idx):
    global global_flash_attn_out_buffer
    global_flash_attn_out_buffer[idx] = None

def write_gradient_to_flash_attn_out(idx, grad):
    global global_flash_attn_out_buffer
    global_flash_attn_out_buffer[idx].grad = grad

def save_res_grad_hook(grad):
    global local_res_grad_buffer
    local_res_grad_buffer = grad

def load_and_add_res_grad_hook(grad):
    grad += get_res_grad_from_local_buffer()

def get_res_grad_from_local_buffer():
    global local_res_grad_buffer
    assert local_res_grad_buffer is not None
    return local_res_grad_buffer

class CheckpointFunctionEndWithFlashAttention(torch.autograd.Function):
    """ Avoid doing twice flash attention forward during checkpointed backward.
    args:
        hidden_states,  # i.e., flash attention output which is saved in global buffer.
        attention_mask,
        position_ids,
        residual,  # the gradient of residual is saved in local buffer to pass across ckpt layers.
    """

    @staticmethod
    def forward(ctx, run_function, layer_idx, preserve_rng_state, *args):
        check_backward_validity(args)
        ctx.run_function = run_function
        ctx.layer_idx = layer_idx
        ctx.preserve_rng_state = preserve_rng_state
        # Accommodates the (remote) possibility that autocast is enabled for cpu AND gpu.
        ctx.gpu_autocast_kwargs, ctx.cpu_autocast_kwargs = _get_autocast_kwargs()
        if preserve_rng_state:
            ctx.fwd_cpu_state = torch.get_rng_state()
            # Don't eagerly initialize the cuda context by accident.
            # (If the user intends that the context is initialized later, within their
            # run_function, we SHOULD actually stash the cuda state here.  Unfortunately,
            # we have no way to anticipate this will happen before we run the function.)
            ctx.had_cuda_in_fwd = False
            if torch.cuda._initialized:
                ctx.had_cuda_in_fwd = True
                ctx.fwd_gpu_devices, ctx.fwd_gpu_states = get_device_states(*args)

        # Save non-tensor inputs in ctx, keep a placeholder None for tensors
        # to be filled out during the backward.
        ctx.inputs = []
        ctx.tensor_indices = []
        tensor_inputs = []
        for i, arg in enumerate(args):
            if i == 0 and ctx.layer_idx != 0:
                # flash attention output is saved to the global buffer during forward
                ctx.inputs.append(None)
            else:
                if torch.is_tensor(arg):
                    tensor_inputs.append(arg)
                    ctx.tensor_indices.append(i)
                    ctx.inputs.append(None)
                else:
                    ctx.inputs.append(arg)

        with torch.no_grad():
            q, k, v, residual = run_function(*args)
            softmax_scale = q.shape[-1] ** (-0.5)

            # lightseq version
            _, _, _, out, softmax_lse = _lightseq_forward(q, k, v, True, softmax_scale, comm_mode='lightseq')
            rng_state = None

            # save flash attention output to global buffer
            save_flash_attn_out_to_global_buffer(ctx.layer_idx, out)
            tensor_inputs += [softmax_lse]
            ctx.softmax_scale = softmax_scale
        
        ctx.save_for_backward(*tensor_inputs)

        return out, residual

    @staticmethod
    def backward(ctx, *args):
        if not torch.autograd._is_checkpoint_valid():
            raise RuntimeError(
                "Checkpointing is not compatible with .grad() or when an `inputs` parameter"
                " is passed to .backward(). Please use .backward() and do not pass its `inputs`"
                " argument.")
        # Copy the list to avoid modifying original list.
        inputs = list(ctx.inputs)
        tensor_indices = ctx.tensor_indices
        tensors = ctx.saved_tensors
        tensors, softmax_lse = tensors[:-1], tensors[-1]

        # Fill in inputs with appropriate saved tensors.
        # Fill the flash attention output first
        if ctx.layer_idx > 0:
            # inputs[0] should be flash attention output
            inputs[0] = get_flash_attn_out_from_global_buffer(ctx.layer_idx-1)
        for i, idx in enumerate(tensor_indices):
            inputs[idx] = tensors[i]

        # Stash the surrounding rng state, and mimic the state that was
        # present at this time during forward.  Restore the surrounding state
        # when we're done.
        rng_devices = []
        if ctx.preserve_rng_state and ctx.had_cuda_in_fwd:
            rng_devices = ctx.fwd_gpu_devices
        with torch.random.fork_rng(devices=rng_devices, enabled=ctx.preserve_rng_state):
            if ctx.preserve_rng_state:
                torch.set_rng_state(ctx.fwd_cpu_state)
                if ctx.had_cuda_in_fwd:
                    set_device_states(ctx.fwd_gpu_devices, ctx.fwd_gpu_states)
            detached_inputs = detach_variable(tuple(inputs))
            with torch.enable_grad(), \
                 torch.cuda.amp.autocast(**ctx.gpu_autocast_kwargs), \
                 torch.cpu.amp.autocast(**ctx.cpu_autocast_kwargs):
                # Stop recomputation before flash attention
                # It is unecessary to run recomputation for flash attn
                q, k, v, residual = ctx.run_function(*detached_inputs)
        
        # run backward() with only tensor that requires grad
        # run flash attention backward first:
        # get 'dout' from auto_grad inputs
        # get 'out' from global buffer
        # get 'qkv' from the recomputed tensors
        #dq = torch.empty(q.shape, dtype=q.dtype, device=q.device)
        #dk = torch.empty(k.shape, dtype=q.dtype, device=q.device)
        #dv = torch.empty(v.shape, dtype=q.dtype, device=q.device)
        out = get_flash_attn_out_from_global_buffer(ctx.layer_idx)
        # todo get dout
        dout = args[0]

        # lightseq version
        dq, dk, dv = _lightseq_backward(dout, q, k, v, out, softmax_lse, ctx.softmax_scale, comm_mode='lightseq', backward_engine='flash')
        #dqkv = torch.stack([dq, dk, dv])

        # run backward for the part before flash attention
        #qkv.backward(dqkv)
        torch.autograd.backward([q, k, v], [dq, dk, dv])

        grads = tuple(inp.grad if isinstance(inp, torch.Tensor) else None
                      for inp in detached_inputs)
        
        # write flash attention output gradients to buffer
        if ctx.layer_idx > 0:
            write_gradient_to_flash_attn_out(ctx.layer_idx-1, detached_inputs[0].grad)

        return (None, None, None) + grads


def checkpoint_end_with_flash_attention(function, layer_idx, *args, use_reentrant: bool = True, **kwargs):
    # Hack to mix *args with **kwargs in a python 2.7-compliant way
    preserve = kwargs.pop('preserve_rng_state', True)
    if kwargs and use_reentrant:
        raise ValueError("Unexpected keyword arguments: " + ",".join(arg for arg in kwargs))

    return CheckpointFunctionEndWithFlashAttention.apply(function, layer_idx, preserve, *args)


class CheckpointFunctionLastModule(torch.autograd.Function):
    """
    for the last ffn layer after flash attention, modifications include:
    write the gradients wrt flash attention output and residual to the global buffer.
    """

    @staticmethod
    def forward(ctx, run_function, preserve_rng_state, *args):
        check_backward_validity(args)
        ctx.run_function = run_function
        ctx.preserve_rng_state = preserve_rng_state
        # Accommodates the (remote) possibility that autocast is enabled for cpu AND gpu.
        ctx.gpu_autocast_kwargs, ctx.cpu_autocast_kwargs = _get_autocast_kwargs()
        if preserve_rng_state:
            ctx.fwd_cpu_state = torch.get_rng_state()
            # Don't eagerly initialize the cuda context by accident.
            # (If the user intends that the context is initialized later, within their
            # run_function, we SHOULD actually stash the cuda state here.  Unfortunately,
            # we have no way to anticipate this will happen before we run the function.)
            ctx.had_cuda_in_fwd = False
            if torch.cuda._initialized:
                ctx.had_cuda_in_fwd = True
                ctx.fwd_gpu_devices, ctx.fwd_gpu_states = get_device_states(*args)

        # Save non-tensor inputs in ctx, keep a placeholder None for tensors
        # to be filled out during the backward.
        ctx.inputs = []
        ctx.tensor_indices = []
        tensor_inputs = []

        assert torch.is_tensor(args[0]), "assuming the first tensor is the flash attention output"
        for i, arg in enumerate(args):
            if torch.is_tensor(arg) and i == 0:
                # flash attn output has been saved to global buffer
                ctx.inputs.append(None)
            elif torch.is_tensor(arg):
                tensor_inputs.append(arg)
                ctx.tensor_indices.append(i)
                ctx.inputs.append(None)
            else:
                ctx.inputs.append(arg)

        ctx.save_for_backward(*tensor_inputs)

        with torch.no_grad():
            outputs = run_function(*args)
        return outputs

    @staticmethod
    def backward(ctx, *args):
        if not torch.autograd._is_checkpoint_valid():
            raise RuntimeError(
                "Checkpointing is not compatible with .grad() or when an `inputs` parameter"
                " is passed to .backward(). Please use .backward() and do not pass its `inputs`"
                " argument.")
        # Copy the list to avoid modifying original list.
        inputs = list(ctx.inputs)
        tensor_indices = ctx.tensor_indices
        tensors = ctx.saved_tensors

        # Fill in inputs with appropriate saved tensors.
        # Fill the flash attention output first
        # inputs[0] should be flash attention output
        inputs[0] = get_flash_attn_out_from_global_buffer(-1)
        for i, idx in enumerate(tensor_indices):
            inputs[idx] = tensors[i]

        # Stash the surrounding rng state, and mimic the state that was
        # present at this time during forward.  Restore the surrounding state
        # when we're done.
        rng_devices = []
        if ctx.preserve_rng_state and ctx.had_cuda_in_fwd:
            rng_devices = ctx.fwd_gpu_devices
        with torch.random.fork_rng(devices=rng_devices, enabled=ctx.preserve_rng_state):
            if ctx.preserve_rng_state:
                torch.set_rng_state(ctx.fwd_cpu_state)
                if ctx.had_cuda_in_fwd:
                    set_device_states(ctx.fwd_gpu_devices, ctx.fwd_gpu_states)
            detached_inputs = detach_variable(tuple(inputs))
            with torch.enable_grad(), \
                 torch.cuda.amp.autocast(**ctx.gpu_autocast_kwargs), \
                 torch.cpu.amp.autocast(**ctx.cpu_autocast_kwargs):
                outputs = ctx.run_function(*detached_inputs)

        if isinstance(outputs, torch.Tensor):
            outputs = (outputs,)

        # run backward() with only tensor that requires grad
        outputs_with_grad = []
        args_with_grad = []
        for i in range(len(outputs)):
            if torch.is_tensor(outputs[i]) and outputs[i].requires_grad:
                outputs_with_grad.append(outputs[i])
                args_with_grad.append(args[i])
        if len(outputs_with_grad) == 0:
            raise RuntimeError(
                "none of output has requires_grad=True,"
                " this checkpoint() is not necessary")
        torch.autograd.backward(outputs_with_grad, args_with_grad)
        grads = tuple(inp.grad if isinstance(inp, torch.Tensor) else None
                      for inp in detached_inputs)
        
        # write flash attention output gradients to buffer
        write_gradient_to_flash_attn_out(-1, detached_inputs[0].grad)

        return (None, None) + grads

def checkpoint_last_module(function, *args, use_reentrant: bool = True, **kwargs):
    preserve = kwargs.pop('preserve_rng_state', True)
    if kwargs and use_reentrant:
        raise ValueError("Unexpected keyword arguments: " + ",".join(arg for arg in kwargs))

    return CheckpointFunctionLastModule.apply(function, preserve, *args)


def llama_layer_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: Optional[bool] = False,
    use_cache: Optional[bool] = False,
    compute_attn_only: Optional[bool] = False,
    compute_ffn_only: Optional[bool] = False,
    residual: Optional[bool] = None,
) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
    """
    Args:
        hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
        attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
            `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under
            returned tensors for more detail.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
            (see `past_key_values`).
        past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
    """
    assert compute_ffn_only or compute_attn_only

    if compute_attn_only:
        residual = hidden_states

        if residual.requires_grad:
            # register a hook to add the gradient of residual 
            # from next checkpoint layer when doing recomputation
            hook = residual.register_hook(load_and_add_res_grad_hook)
            global_hooks.append(hook)

        hidden_states = self.input_layernorm(hidden_states)

        # Flash Attention
        bsz, q_len, _ = hidden_states.size()
        try:
            query_states = self.self_attn.q_proj(hidden_states).view(bsz, q_len, self.self_attn.num_heads, self.self_attn.head_dim).transpose(1, 2)
            key_states = self.self_attn.k_proj(hidden_states).view(bsz, q_len, self.self_attn.num_key_value_heads, self.self_attn.head_dim).transpose(1, 2)
            value_states = self.self_attn.v_proj(hidden_states).view(bsz, q_len, self.self_attn.num_key_value_heads, self.self_attn.head_dim).transpose(1, 2)
        except:
            # old transformers versions don't support num_key_value_heads
            query_states = self.self_attn.q_proj(hidden_states).view(bsz, q_len, self.self_attn.num_heads, self.self_attn.head_dim).transpose(1, 2)
            key_states = self.self_attn.k_proj(hidden_states).view(bsz, q_len, self.self_attn.num_heads, self.self_attn.head_dim).transpose(1, 2)
            value_states = self.self_attn.v_proj(hidden_states).view(bsz, q_len, self.self_attn.num_heads, self.self_attn.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        assert past_key_value is None, "past_key_value is not supported"

        cos, sin = self.self_attn.rotary_emb(value_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        # [bsz, nh, t, hd]
        assert not output_attentions, "output_attentions is not supported"
        assert not use_cache, "use_cache is not supported"
        return query_states.contiguous(), key_states.contiguous(), value_states.contiguous(), residual

    elif compute_ffn_only:
        hidden_states = self.self_attn.o_proj(rearrange(hidden_states, 'b h s d -> b s (h d)'))
        # Need to add residual here to make sure checkpoint is right after attention
        if residual.requires_grad:
            # save the gradient of residual to the local buffer
            # collect the hooks which should be removed after backward to avoid memory leak
            hook = residual.register_hook(save_res_grad_hook)
            global_hooks.append(hook)
        
        hidden_states = residual + hidden_states

        # Fully Connected

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

    else:
        raise AttributeError

    return outputs


def forward(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[List[torch.FloatTensor]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    cache_position: Optional[torch.LongTensor] = None,
    return_dict: Optional[bool] = None,
):  
    assert cache_position is None, "cache_position is not supported"
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    use_cache = use_cache if use_cache is not None else self.config.use_cache

    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    # retrieve input_ids and inputs_embeds
    if input_ids is not None and inputs_embeds is not None:
        raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
    elif input_ids is not None:
        batch_size, seq_length = input_ids.shape
    elif inputs_embeds is not None:
        batch_size, seq_length, _ = inputs_embeds.shape
    else:
        raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

    seq_length_with_past = seq_length
    past_key_values_length = 0

    if past_key_values is not None:
        past_key_values_length = past_key_values[0][0].shape[2]
        seq_length_with_past = seq_length_with_past + past_key_values_length

    if position_ids is None:
        device = input_ids.device if input_ids is not None else inputs_embeds.device
        position_ids = torch.arange(
            past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
        )
        position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
    else:
        position_ids = position_ids.view(-1, seq_length).long()

    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)
    # embed positions
    attention_mask = None

    hidden_states = inputs_embeds

    if self.gradient_checkpointing and self.training:
        try:
            logger.warning_once(
                "***** Using fast gradient checkpointing... *****"
            )
        except:
            pass
        # initialize the global buffer
        init_flash_attn_buffers(len(self.layers))

        if use_cache:
            try:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
            except:
                pass
            use_cache = False

    # decoder layers
    all_hidden_states = () if output_hidden_states else None
    all_self_attns = () if output_attentions else None
    next_decoder_cache = () if use_cache else None

    # apply flash-attention friendly gradient checkpointing
    if self.gradient_checkpointing and self.training:
        for idx in range(len(self.layers) + 1):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            
            past_key_value = past_key_values[idx] if past_key_values is not None else None

            def forward_first_attn_module(module):
                def custom_forward(*inputs):
                    hidden_states, attention_mask, position_ids, _ = inputs
                    # None for past_key_value
                    return module(hidden_states, attention_mask, position_ids, past_key_value, output_attentions, compute_attn_only=True)
                return custom_forward
            
            def forward_ffn_attn_layer(module1, module2):
                def custom_forward(*inputs):
                    hidden_states, attention_mask, position_ids, residual = inputs
                    # None for past_key_value
                    layer_outputs = module1(hidden_states, attention_mask, position_ids, past_key_value, output_attentions, compute_ffn_only=True, residual=residual)
                    hidden_states = layer_outputs[0]
                    return module2(hidden_states, attention_mask, position_ids, past_key_value, output_attentions, compute_attn_only=True)
                return custom_forward
            
            def forward_last_ffn_module(module):
                def custom_forward(*inputs):
                    hidden_states, attention_mask, position_ids, residual = inputs
                    # None for past_key_value
                    return module(hidden_states, attention_mask, position_ids, past_key_value, output_attentions, compute_ffn_only=True, residual=residual)
                return custom_forward
            
            if idx == 0:
                layer_outputs = checkpoint_end_with_flash_attention(
                    forward_first_attn_module(self.layers[0]),
                    idx,
                    hidden_states,
                    attention_mask,
                    position_ids,
                    None,
                )
                hidden_states, residual = layer_outputs[0], layer_outputs[-1]
            elif idx == len(self.layers):
                layer_outputs = checkpoint_last_module(
                    forward_last_ffn_module(self.layers[-1]),
                    hidden_states,
                    attention_mask,
                    position_ids,
                    residual,
                )
                hidden_states = layer_outputs[0]
            else:
                layer_outputs = checkpoint_end_with_flash_attention(
                    forward_ffn_attn_layer(self.layers[idx-1], self.layers[idx]),
                    idx,
                    hidden_states,
                    attention_mask,
                    position_ids,
                    residual,
                )
                hidden_states, residual = layer_outputs[0], layer_outputs[-1]

            if use_cache:
                next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)
    else:
        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            past_key_value = past_key_values[idx] if past_key_values is not None else None

            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

    hidden_states = self.norm(hidden_states)

    # add hidden states from the last decoder layer
    if output_hidden_states:
        all_hidden_states += (hidden_states,)

    next_cache = next_decoder_cache if use_cache else None
    if not return_dict:
        return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
    return BaseModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=next_cache,
        hidden_states=all_hidden_states,
        attentions=all_self_attns,
    )


def apply_dist_flash_attn_monkey_patch_llama(sp_size=None):
    initialize_distributed(sp_size=sp_size)
    transformers.models.llama.modeling_llama.LlamaModel.forward = forward
    transformers.models.llama.modeling_llama.LlamaDecoderLayer.forward = llama_layer_forward
