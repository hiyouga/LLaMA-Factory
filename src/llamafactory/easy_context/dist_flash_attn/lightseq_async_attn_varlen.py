import os
import math

from einops import rearrange
import argparse

import pytest
import torch
import torch.distributed as dist
from torch.distributed import ReduceOp
#from torch.profiler import profile, record_function, ProfilerActivity

import triton
import triton.language as tl
import time
import numpy as np
from tqdm import tqdm

try:
    from flash_attn.flash_attn_interface import _flash_attn_varlen_backward
except:
    pass

from .async_communication import (is_last_time, is_compute_for_local_query, is_sync_from_remote, is_idle, print_and_reset_comm_stats, 
        launch_async_handles, wait_async_handles, maybe_send_recv_fwd_qkvo, maybe_send_recv_bwd_qkvo, maybe_send_recv_bwd_last_dkv, reset_global_memory_buffer,
        maybe_get_set_global_memory_buffer, maybe_get_set_global_memory_buffer_bwd, initialize_distributed, get_sequence_parallel_size, get_sequence_parallel_rank)

@triton.jit
def max_fn(x, y):
    return tl.math.max(x, y)

@triton.jit
def _rescale_kernel(
    peer_m,
    m,
    peer_l,
    l,
    peer_o,
    o,
    L,
    stride_oz, stride_oh, stride_om, stride_on,
    Z, H, N_CTX,
    seqlen_q_rounded, seqlen_peer_q_rounded,
    BLOCK_M: tl.constexpr, BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
    LAST_STEP: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    o_offset = off_hz * stride_oh
    peer_o_block_ptr = tl.make_block_ptr(
        base=peer_o + o_offset,
        shape=(N_CTX, BLOCK_DMODEL),
        strides=(stride_om, stride_on),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL),
        order=(1, 0)
    )
    o_block_ptr = tl.make_block_ptr(
        base=o + o_offset,
        shape=(N_CTX, BLOCK_DMODEL),
        strides=(stride_om, stride_on),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL),
        order=(1, 0)
    )
    # initialize offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)

    peer_m_ptrs = peer_m + off_hz * seqlen_peer_q_rounded + offs_m
    m_ptrs = m + off_hz * seqlen_q_rounded + offs_m
    peer_l_ptrs = peer_l + off_hz * seqlen_peer_q_rounded + offs_m
    l_ptrs = l + off_hz * seqlen_q_rounded + offs_m
    
    peer_m_i = tl.load(peer_m_ptrs) 
    peer_m_i = peer_m_i.to(tl.float32)
    m_i = tl.load(m_ptrs) 
    m_i = m_i.to(tl.float32)
    peer_l_i = tl.load(peer_l_ptrs) 
    peer_l_i = peer_l_i.to(tl.float32)
    l_i = tl.load(l_ptrs) 
    l_i = l_i.to(tl.float32)

    peer_acc = tl.load(peer_o_block_ptr)#, boundary_check=(0, 1), padding_option='zero')
    peer_acc = peer_acc.to(tl.float32)
    acc = tl.load(o_block_ptr) #, boundary_check=(0, 1), padding_option='zero') 
    acc = acc.to(tl.float32)
    lo = 0
    hi = N_CTX
    m_i_sync = tl.maximum(m_i, peer_m_i)
    alpha = tl.math.exp2(m_i - m_i_sync)
    peer_alpha = tl.math.exp2(peer_m_i - m_i_sync)
    # -- scale and update acc --
    acc_scale = l_i * 0 + alpha  # workaround some compiler bug
    peer_acc_scale = peer_l_i * 0 + peer_alpha  # workaround some compiler bug
    
    acc *= acc_scale[:, None]
    peer_acc *= peer_acc_scale[:, None]
    acc += peer_acc
    l_i = l_i * acc_scale + peer_l_i * peer_acc_scale
    # write back O, l, m
    tl.store(m_ptrs, m_i_sync)
    tl.store(l_ptrs, l_i)
    if LAST_STEP:
        acc = acc / l_i[:, None]
        L_ptrs = L + off_hz * N_CTX + offs_m
        tl.store(L_ptrs, m_i_sync / 1.44269504 + tl.math.log(l_i))
    tl.store(o_block_ptr, acc.to(tl.bfloat16), boundary_check=(0, 1))

@triton.jit
def _fwd_kernel(
    Q, K, V, sm_scale,
    m,
    l,
    O,
    L,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vk, stride_vn,
    stride_oz, stride_oh, stride_om, stride_on,
    Z, H, N_CTX,
    seqlen_q_rounded,
    BLOCK_M: tl.constexpr, BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    LAST_STEP: tl.constexpr
):
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    qvk_offset = off_hz * stride_qh
    Q_block_ptr = tl.make_block_ptr(
        base=Q + qvk_offset,
        shape=(N_CTX, BLOCK_DMODEL),
        strides=(stride_qm, stride_qk),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL),
        order=(1, 0)
    )
    K_block_ptr = tl.make_block_ptr(
        base=K + qvk_offset,
        shape=(BLOCK_DMODEL, N_CTX),
        strides=(stride_kk, stride_kn),
        offsets=(0, 0),
        block_shape=(BLOCK_DMODEL, BLOCK_N),
        order=(0, 1)
    )
    V_block_ptr = tl.make_block_ptr(
        base=V + qvk_offset,
        shape=(N_CTX, BLOCK_DMODEL),
        strides=(stride_vk, stride_vn),
        offsets=(0, 0),
        block_shape=(BLOCK_N, BLOCK_DMODEL),
        order=(1, 0)
    )
    O_block_ptr = tl.make_block_ptr(
        base=O + qvk_offset,
        shape=(N_CTX, BLOCK_DMODEL),
        strides=(stride_om, stride_on),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL),
        order=(1, 0)
    )
    # initialize offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    # initialize pointer to m and l -> load from provided pointer
    # (TODO): Why float32?
    m_ptrs = m + off_hz * seqlen_q_rounded + offs_m
    l_ptrs = l + off_hz * seqlen_q_rounded + offs_m
    m_i = tl.load(m_ptrs) 
    m_i = m_i.to(tl.float32)
    l_i = tl.load(l_ptrs) 
    l_i = l_i.to(tl.float32)
    acc = tl.load(O_block_ptr) 
    acc = acc.to(tl.float32)
    # scale sm_scale by log_2(e) and use
    # 2^x instead of exp in the loop because CSE and LICM
    # don't work as expected with `exp` in the loop
    qk_scale = sm_scale * 1.44269504
    # load q: it will stay in SRAM throughout
    q = tl.load(Q_block_ptr, boundary_check=(0,), padding_option='zero')
    q = (q * qk_scale).to(tl.bfloat16)
    # loop over k, v and update accumulator
    lo = 0
    hi = (start_m + 1) * BLOCK_M if IS_CAUSAL else N_CTX
    for start_n in range(lo, hi, BLOCK_N):
        # -- load k, v --
        k = tl.load(K_block_ptr, boundary_check=(1,), padding_option='zero')
        v = tl.load(V_block_ptr, boundary_check=(0,), padding_option='zero')
        # -- compute qk ---
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        if IS_CAUSAL:
            qk = tl.where(offs_m[:, None] >= (start_n + offs_n[None, :]), qk, float("-inf"))
        qk += tl.dot(q, k)
        # -- compute scaling constant ---
        m_i_new = tl.maximum(m_i, tl.max(qk, 1))
        alpha = tl.math.exp2(m_i - m_i_new)
        p = tl.math.exp2(qk - m_i_new[:, None])
        # -- scale and update acc --
        acc_scale = l_i * 0 + alpha  # workaround some compiler bug
        acc *= acc_scale[:, None]
        acc += tl.dot(p.to(tl.bfloat16), v)
        # -- update m_i and l_i --
        l_i = l_i * alpha + tl.sum(p, 1)
        m_i = m_i_new
        # update pointers
        K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_N))
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))
    # write back original l and m
    tl.store(m_ptrs, m_i)
    tl.store(l_ptrs, l_i)
    # write back O, L
    if LAST_STEP:
        acc = acc / l_i[:, None]
        L_ptrs = L + off_hz * seqlen_q_rounded + offs_m
        tl.store(L_ptrs, m_i / 1.44269504 + tl.math.log(l_i))
    tl.store(O_block_ptr, acc.to(tl.bfloat16), boundary_check=(0, 1))

# for gqa/mqa to expand kv heads
def maybe_repeat_kv_fwd(nqh, kv):
    bs, nkvh, slen, hdim = kv.shape
    n_rep = nqh // nkvh
    if n_rep == 1:
        return kv
    kv_expand = kv[:, :, None, :, :].expand(bs, nkvh, n_rep, slen, hdim)
    return kv_expand.reshape(bs, nkvh * n_rep, slen, hdim)

def maybe_repeat_kv_bwd(nqh, kv):
    bs, slen, nkvh, hdim = kv.shape
    n_rep = nqh // nkvh
    if n_rep == 1:
        return kv
    kv_expand = kv[:, :, :, None, :].expand(bs, slen, nkvh, n_rep, hdim)
    return kv_expand.reshape(bs, slen, nkvh * n_rep, hdim)

# kv grad has shape bs, slen, nqh, hdim
def maybe_reduce_dkv(nkvh, dkv):
    bs, slen, nqh, hdim = dkv.shape
    n_rep = nqh // nkvh
    if n_rep == 1:
        return dkv
    #print("*"*100, dkv.shape, bs, slen, nkvh, n_rep, hdim)
    dkv_reshape = dkv.view(bs, slen, nkvh, n_rep, hdim)
    #print("-"*100, dkv_reshape.shape, bs, slen, nkvh, n_rep, hdim)
    return torch.sum(dkv_reshape, dim=3)


def _lightseq_forward_varlen(q, k, v, causal, sm_scale, comm_mode):
    # maybe_contiguous = lambda x: x.contiguous() if x.stride(-1) != 1 else x
    # q, k, v = [maybe_contiguous(x) for x in (q, k, v)]

    # shape constraints
    Lq, Lk, Lv = q.shape[-1], k.shape[-1], v.shape[-1]
    # assert Lq == Lk and Lk == Lv
    # assert Lk in {16, 32, 64, 128}
    BLOCK_M = 128
    BLOCK_N = 64

    bsz, nh, unpadded_seq_len, hdim = q.shape
    cu_seq_lens = torch.arange(0, (bsz+1) * unpadded_seq_len, unpadded_seq_len, dtype=torch.int32, device=q.device)
    max_seqlen = unpadded_seq_len
    seqlen_q_rounded = math.ceil(q.shape[2] / BLOCK_M) * BLOCK_M

    m = torch.full((bsz * nh, seqlen_q_rounded), fill_value=-float("inf"), device=q.device, dtype=torch.float32)
    l = torch.zeros((bsz * nh, seqlen_q_rounded), device=q.device, dtype=torch.float32)
    L = torch.zeros((bsz * nh, seqlen_q_rounded), device=q.device, dtype=torch.float32)
    o = torch.zeros_like(q)
    
    grid = (triton.cdiv(q.shape[2], BLOCK_M), bsz * nh, 1)
    num_warps = 4 if Lk <= 64 else 8
    
    seq_rank = get_sequence_parallel_rank()
    seq_world_size = get_sequence_parallel_size()

    # Initialize all buffers
    peer_q, peer_k, peer_v, peer_m, peer_l, peer_o = maybe_get_set_global_memory_buffer(q, k, v, m, l, o)
    
    fwd_launch_helper = lambda q, k, v, m, l, o, L, IS_CAUSAL, LAST_STEP: _fwd_kernel[grid](
                q, k, v, sm_scale,
                m,
                l,
                o,
                L,
                q.stride(0), q.stride(1), q.stride(2), q.stride(3),
                k.stride(0), k.stride(1), k.stride(2), k.stride(3),
                v.stride(0), v.stride(1), v.stride(2), v.stride(3),
                o.stride(0), o.stride(1), o.stride(2), o.stride(3),
                q.shape[0], q.shape[1], q.shape[2],
                seqlen_q_rounded,
                BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_DMODEL=Lk,
                IS_CAUSAL=IS_CAUSAL,
                LAST_STEP=LAST_STEP,
                num_warps=num_warps,
                num_stages=4)
    
    for time_step in range(seq_world_size // 2 + 1):
        # This is important for cuda scheduler to execute nccl calls first.
        torch.cuda.synchronize()
        # Communication uses buffer_idx_1, and compute uses buffer_idx_2, which effectively are contents from the last time step.
        buffer_idx_1 = time_step % 2
        buffer_idx_2 = (time_step - 1) % 2

        reqs = maybe_send_recv_fwd_qkvo(q, peer_q[buffer_idx_1], k, peer_k[buffer_idx_1], v, peer_v[buffer_idx_1], 
                                           [peer_o[buffer_idx_1], peer_m[buffer_idx_1], peer_l[buffer_idx_1]], time_step, comm_mode)
        if comm_mode == "sync":
            # if seq_rank == 0:
            #    print("Immediate wait for abalation")
            wait_async_handles(reqs)
        if is_compute_for_local_query(time_step):
            # print(f"t={time_step}: (Comp) R={seq_rank} local compute")
            if time_step == 0:
                fwd_launch_helper(q, maybe_repeat_kv_fwd(q.shape[1], k), maybe_repeat_kv_fwd(q.shape[1], v), m, l, o, L, True, is_last_time(time_step))
            else:
                # if needs to sync from others, do not normalize here
                fwd_launch_helper(q, maybe_repeat_kv_fwd(q.shape[1], peer_k[buffer_idx_2]), maybe_repeat_kv_fwd(q.shape[1], peer_v[buffer_idx_2]), m, l, o, L, False, not is_sync_from_remote(time_step) and is_last_time(time_step))
        elif is_idle(time_step):
            # print(f"t={time_step}: (Comp) R={seq_rank} idle")
            pass
        else:
            # print(f"t={time_step}: (Comp) R={seq_rank} helps other")
            peer_m[buffer_idx_2] = torch.full_like(m, fill_value=-float("inf"))
            peer_l[buffer_idx_2] = torch.zeros_like(l)
            peer_o[buffer_idx_2] = torch.zeros_like(o)

            #print(f"rank 3 q is: {peer_q[buffer_idx_2]}")
            fwd_launch_helper(peer_q[buffer_idx_2], maybe_repeat_kv_fwd(q.shape[1], k), maybe_repeat_kv_fwd(q.shape[1], v), peer_m[buffer_idx_2], peer_l[buffer_idx_2], peer_o[buffer_idx_2], None, False, False)

        if comm_mode == "lightseq":
            # Make sure tensors for next steps are ready
            wait_async_handles(reqs)
        # sync between statistics get from other ranks and the local ones
        if is_sync_from_remote(time_step):
#             print(f"t={time_step}: (Comp) R={seq_rank} sync with other - last time: {is_last_time(time_step)}")
            seqlen_peer_q_rounded = peer_l[buffer_idx_1].shape[-1]
            _rescale_kernel[grid](
                peer_m[buffer_idx_1],
                m,
                peer_l[buffer_idx_1],
                l,
                peer_o[buffer_idx_1],
                o,
                L,
                o.stride(0), o.stride(1), o.stride(2), o.stride(3),
                o.shape[0], o.shape[1], o.shape[2],
                seqlen_q_rounded, seqlen_peer_q_rounded,
                BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_DMODEL=Lk,
                LAST_STEP=is_last_time(time_step),
                num_warps=num_warps,
                num_stages=4)
    return q, k, v, o, L, cu_seq_lens, max_seqlen

def _lightseq_backward_varlen(do, q, k, v, o, L, sm_scale, comm_mode, backward_engine, cu_seq_lens, max_seqlen):
    BLOCK = 128
    L = rearrange(L[:, :max_seqlen].contiguous(), '(b h) s -> b h s', b=q.shape[0])
    q, k, v, o, do = [rearrange(_x, 'b h s d -> (b s) h d').contiguous() for _x in [q, k, v, o, do]]
    
    dq = torch.empty_like(q)
    dk = torch.empty_like(k)
    dv = torch.empty_like(v)

    # maybe gqa
    nqh = q.shape[1]
    nkvh = k.shape[1]
    is_gqa = (nqh > nkvh)

    seq_rank = get_sequence_parallel_rank()
    seq_world_size = get_sequence_parallel_size()
   
    # Initialize all backward buffers
    dq_delta, dk_delta, dv_delta, dk_delta_from_peer, dv_delta_from_peer, \
            peer_q, peer_L, peer_k, peer_v, peer_o, peer_do = maybe_get_set_global_memory_buffer_bwd(dq, dk, dv, q, L, k, v, o, do)
    
    for time_step in range(0, get_sequence_parallel_size() // 2 + 1):
        torch.cuda.synchronize()
        buffer_idx_1 = time_step % 2
        buffer_idx_2 = (time_step - 1) % 2
        
        reqs, is_update_dq, is_update_dkv = maybe_send_recv_bwd_qkvo(dq_delta[buffer_idx_1], dk_delta[buffer_idx_1], dv_delta[buffer_idx_1], dk_delta_from_peer, dv_delta_from_peer, q, peer_q[buffer_idx_1], L, peer_L[buffer_idx_1], k, peer_k[buffer_idx_1], v, peer_v[buffer_idx_1], o, peer_o[buffer_idx_1], do, peer_do[buffer_idx_1], time_step, comm_mode)
        if comm_mode == "sync":
            wait_async_handles(reqs)

        if is_compute_for_local_query(time_step):
            if time_step == 0:
                assert backward_engine == "flash", "We haven't supportted varlen feature in xformer"
                if backward_engine == "flash":
                    _flash_attn_varlen_backward(do, q, k, v, o, L, dq, dk, dv, cu_seq_lens, cu_seq_lens, max_seqlen, max_seqlen, 0.0, sm_scale, True, None)
                else:
                    inp = Inputs(query=q, key=maybe_repeat_kv_bwd(q.shape[2], k), value=maybe_repeat_kv_bwd(q.shape[2], v), attn_bias=xformers.ops.LowerTriangularMask(), p=0, scale=sm_scale)
                    op_ctx = Context(lse=L, out=o, rng_state=None)
                    # Let xformers dispatch the correct backend
                    grads = _memory_efficient_attention_backward(ctx=op_ctx, inp=inp, grad=do, op=None)
                    dq = grads.dq
                    dk, dv = maybe_reduce_dkv(nkvh, grads.dk), maybe_reduce_dkv(nkvh, grads.dv)
            else:
                assert backward_engine == "flash", "We haven't supportted varlen feature in xformer"
                if backward_engine == "flash":
                    _flash_attn_varlen_backward(do, q, peer_k[buffer_idx_2], peer_v[buffer_idx_2], o, L, dq_delta[buffer_idx_2], dk_delta[buffer_idx_2], dv_delta[buffer_idx_2], cu_seq_lens, cu_seq_lens, max_seqlen, max_seqlen, 0.0, sm_scale, False, None)
                else:
                    inp = Inputs(query=q, key=maybe_repeat_kv_bwd(q.shape[2], peer_k[buffer_idx_2]), value=maybe_repeat_kv_bwd(q.shape[2], peer_v[buffer_idx_2]), attn_bias=None, p=0, scale=sm_scale)
                    op_ctx = Context(lse=L, out=o, rng_state=None)
                    grads = _memory_efficient_attention_backward(ctx=op_ctx, inp=inp, grad=do, op=None)
                    dq_delta[buffer_idx_2] = grads.dq
                    dk_delta[buffer_idx_2], dv_delta[buffer_idx_2] = maybe_reduce_dkv(nkvh, grads.dk), maybe_reduce_dkv(nkvh, grads.dv)
                dq += dq_delta[buffer_idx_2]
        elif is_idle(time_step):
        #    print(f"BWD t={time_step}: (Comp) R={seq_rank} idle")
            pass
        else:
        #    print(f"BWD t={time_step}: (Comp) R={seq_rank} helps other")
            assert backward_engine == "flash", "We haven't supportted varlen feature in xformer"
            if backward_engine == "flash":
                _flash_attn_varlen_backward(peer_do[buffer_idx_2], peer_q[buffer_idx_2], k, v, peer_o[buffer_idx_2], peer_L[buffer_idx_2], dq_delta[buffer_idx_2], dk_delta[buffer_idx_2], dv_delta[buffer_idx_2], cu_seq_lens, cu_seq_lens, max_seqlen, max_seqlen, 0.0, sm_scale, False, None)
            else:
                inp = Inputs(query=peer_q[buffer_idx_2], key=maybe_repeat_kv_bwd(q.shape[2], k), value=maybe_repeat_kv_bwd(q.shape[2], v), attn_bias=None, p=0, scale=sm_scale)
                op_ctx = Context(lse=peer_L[buffer_idx_2], out=peer_o[buffer_idx_2], rng_state=None)
                grads = _memory_efficient_attention_backward(ctx=op_ctx, inp=inp, grad=peer_do[buffer_idx_2], op=None)
                dq_delta[buffer_idx_2] = grads.dq
                dk_delta[buffer_idx_2], dv_delta[buffer_idx_2] = maybe_reduce_dkv(nkvh, grads.dk), maybe_reduce_dkv(nkvh, grads.dv)
            dk += dk_delta[buffer_idx_2]
            dv += dv_delta[buffer_idx_2]
        
        if comm_mode == "lightseq":
            # Make sure tensors for next steps are ready
            wait_async_handles(reqs)
        
        # The last time step needs to send dk and dv immediately, move it up here to maximize overlap with the following three addition.
        reqs, is_update_last_dkv = maybe_send_recv_bwd_last_dkv(dk_delta[buffer_idx_2], dv_delta[buffer_idx_2], time_step, comm_mode)
        
        if comm_mode == "sync":
            # if seq_rank == 0:
            #    print("(bwd) dkv Immediate wait for abalation")
            wait_async_handles(reqs)
        # apply dq_delta, dk_delta and dv_delta from remote
        if is_update_dq:
            dq += dq_delta[buffer_idx_1]
        if is_update_dkv:
            dk += dk_delta_from_peer
            dv += dv_delta_from_peer
       
        if comm_mode == "lightseq":
            wait_async_handles(reqs)
        # apply dk_delta and dv_delta to sender
        if is_update_last_dkv:
            dk += dk_delta[buffer_idx_2]
            dv += dv_delta[buffer_idx_2]
                
    dq, dk, dv = [rearrange(_x, '(b s) h d -> b h s d', s=max_seqlen) for _x in [dq, dk, dv]]
    return dq, dk, dv

class _attention_varlen(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, causal, sm_scale):
        try:
            global args
            comm_mode = args.comm_mode
            backward_engine = args.backward_engine
        except:
            comm_mode = 'lightseq'
            backward_engine = 'flash'
        
        q, k, v, o, L, cu_seq_lens, max_seqlen = _lightseq_forward_varlen(q, k, v, causal, sm_scale, comm_mode)

        ctx.save_for_backward(q, k, v, o, L, cu_seq_lens)
        ctx.max_seqlen = max_seqlen
        ctx.sm_scale = sm_scale
        ctx.comm_mode = comm_mode
        ctx.backward_engine = backward_engine
        return o

    @staticmethod
    def backward(ctx, do):
        q, k, v, o, L, cu_seq_lens = ctx.saved_tensors 
        sm_scale = ctx.sm_scale
        max_seqlen = ctx.max_seqlen

        dq, dk, dv = _lightseq_backward_varlen(do, q, k, v, o, L, sm_scale, ctx.comm_mode, ctx.backward_engine, cu_seq_lens, max_seqlen)
        return dq, dk, dv, None, None

dist_attn_varlen = _attention_varlen.apply


#@pytest.mark.parametrize('causal', [False, True])
#@pytest.mark.parametrize('Z, H, N_CTX, D_HEAD', [(6, 9, 1024, 64)])
def test_op(Z, H, N_CTX, D_HEAD, causal, dtype=torch.bfloat16):
    torch.manual_seed(20)
    rank = dist.get_rank()
    world_size = dist.get_world_size()
   

    PAD = world_size * 256
    seq_per_rank = (N_CTX-PAD) // world_size
    q = torch.empty((Z, H, N_CTX-PAD, D_HEAD), dtype=dtype, device="cuda").normal_(mean=0., std=0.5).requires_grad_()
    k = torch.empty((Z, H, N_CTX-PAD, D_HEAD), dtype=dtype, device="cuda").normal_(mean=0., std=0.5).requires_grad_()
    v = torch.empty((Z, H, N_CTX-PAD, D_HEAD), dtype=dtype, device="cuda").normal_(mean=0., std=0.5).requires_grad_()
    
    # DEBUG: mask out
    #mask = torch.zeros(Z, H, seq_per_rank * (world_size - 1), D_HEAD).cuda()
    #mask_2 = torch.ones(Z, H, seq_per_rank, D_HEAD).cuda()
    #mask = torch.cat((mask, mask_2), dim=-2).to(dtype)
    #q = mask * q
    #k = mask * k
    #v = mask * v

    sm_scale = 0.5
    dout = torch.randn_like(q)
    # reference implementation
    M = torch.tril(torch.ones((N_CTX-PAD, N_CTX-PAD), device="cuda"))
    p = torch.matmul(q, k.transpose(2, 3)) * sm_scale
    assert causal
    if causal:
        p[:, :, M == 0] = float("-inf")
    p = torch.softmax(p.float(), dim=-1).half()
    ref_out = torch.matmul(p, v)
    ref_out.backward(dout)
    ref_dv, v.grad = v.grad.clone(), None
    ref_dk, k.grad = k.grad.clone(), None
    ref_dq, q.grad = q.grad.clone(), None

    # triton implementation
   
    a, b, c, d = q.size()
    real_q = q[:,:, rank * seq_per_rank: (rank + 1) * seq_per_rank, :].view(a, b, -1, d).contiguous().clone().detach().requires_grad_(True)
    real_k = k[:,:, rank * seq_per_rank: (rank + 1) * seq_per_rank, :].view(a, b, -1, d).contiguous().clone().detach().requires_grad_(True)
    real_v = v[:,:, rank * seq_per_rank: (rank + 1) * seq_per_rank, :].view(a, b, -1, d).contiguous().clone().detach().requires_grad_(True)
    real_do = dout[:,:, rank * seq_per_rank: (rank + 1) * seq_per_rank, :].view(a, b, -1, d).contiguous().clone().detach().requires_grad_(True)
    
    tri_out = dist_attn_varlen(real_q, real_k, real_v, causal, sm_scale).half()

    # compare
    assert torch.allclose(ref_out[:, :, rank * seq_per_rank: (rank + 1) * seq_per_rank, :], tri_out, atol=1e-2, rtol=0), f" rank {rank} fails forward"
    print(f" *** rank {rank} passes forward")
    tri_out.backward(real_do)
    tri_dv, real_v.grad = real_v.grad.clone(), None
    tri_dk, real_k.grad = real_k.grad.clone(), None
    tri_dq, real_q.grad = real_q.grad.clone(), None
    assert torch.allclose(ref_dq[:, :, rank * seq_per_rank: (rank + 1) * seq_per_rank, :], tri_dq, atol=1e-2, rtol=0),  f"rank {rank} fails backward dq" #{ref_dq[:, :, rank * seq_per_rank: (rank + 1) * seq_per_rank, :]} {tri_dq} {torch.max(ref_dq[:, :, rank * seq_per_rank: (rank + 1) * seq_per_rank, :] - tri_dq)} rank {rank} fails backward dk"
    assert torch.allclose(ref_dk[:, :, rank * seq_per_rank: (rank + 1) * seq_per_rank, :], tri_dk, atol=1e-2, rtol=0),  f"rank {rank} fails backward dk" #{ref_dk[:, :, rank * seq_per_rank: (rank + 1) * seq_per_rank, :]} {tri_dk} {torch.max(ref_dk[:, :, rank * seq_per_rank: (rank + 1) * seq_per_rank, :] - tri_dk)} rank {rank} fails backward dk"
    assert torch.allclose(ref_dv[:, :, rank * seq_per_rank: (rank + 1) * seq_per_rank, :], tri_dv, atol=1e-2, rtol=0),  f"rank {rank} fails backward dv" #{ref_dv[:, :, rank * seq_per_rank: (rank + 1) * seq_per_rank, :]} {tri_dv} {torch.max(ref_dv[:, :, rank * seq_per_rank: (rank + 1) * seq_per_rank, :] - tri_dv)} rank {rank} fails backward dv"
    print(f"rank {rank} passes backward")

#TODO(High Priority): Investigate why rank 0 tends to have larger numerical difference.
def test_gqa(Z, H, KVH, N_CTX, D_HEAD, causal, dtype=torch.bfloat16):
    torch.manual_seed(177)
    q = torch.empty((Z, H, N_CTX, D_HEAD), dtype=dtype, device="cuda").normal_(mean=0., std=0.5).requires_grad_()
    k = torch.empty((Z, KVH, N_CTX, D_HEAD), dtype=dtype, device="cuda").normal_(mean=0., std=0.5).requires_grad_()
    v = torch.empty((Z, KVH, N_CTX, D_HEAD), dtype=dtype, device="cuda").normal_(mean=0., std=0.5).requires_grad_()
    
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    seq_per_rank = N_CTX // world_size

    sm_scale = 0.5
    dout = torch.randn_like(q)
    # torch reference implementation
    M = torch.tril(torch.ones((N_CTX, N_CTX), device="cuda"))
    ref_k = maybe_repeat_kv_fwd(q.shape[1], k).clone().detach().requires_grad_(True)
    ref_v = maybe_repeat_kv_fwd(q.shape[1], v).clone().detach().requires_grad_(True)
    #print(q.shape, ref_k.shape, k.shape)
    p = torch.matmul(q, ref_k.transpose(2,3)) * sm_scale
    assert causal
    if causal:
        p[:, :, M == 0] = float("-inf")
    p = torch.softmax(p.float(), dim=-1).half()
    ref_out = torch.matmul(p, ref_v)
    ref_out.backward(dout)
    ref_dv, v.grad = ref_v.grad.clone(), None
    #print("Before reduce", ref_dv.shape)
    ref_dv = (maybe_reduce_dkv(KVH, ref_dv.transpose(1,2))).transpose(1,2)
    #print("After reduce", ref_dv.shape)
    ref_dk, k.grad = ref_k.grad.clone(), None
    ref_dk = (maybe_reduce_dkv(KVH, ref_dk.transpose(1,2))).transpose(1,2)
    ref_dq, q.grad = q.grad.clone(), None

    # flash reference
    from flash_attn import flash_attn_qkvpacked_func, flash_attn_func
    flash_q = q.transpose(1,2).clone().detach().requires_grad_(True)
    flash_k = k.transpose(1,2).clone().detach().requires_grad_(True)
    flash_v = v.transpose(1,2).clone().detach().requires_grad_(True)
    flash_ref_out = flash_attn_func(flash_q, flash_k, flash_v, 0, sm_scale, True)
    flash_ref_out.backward(dout.transpose(1,2))
    flash_ref_out = flash_ref_out.transpose(1,2)
    flash_ref_dv, v.grad = flash_v.grad.clone(), None
    flash_ref_dv = flash_ref_dv.transpose(1,2)
    flash_ref_dk, k.grad = flash_k.grad.clone(), None
    flash_ref_dk = flash_ref_dk.transpose(1,2)
    flash_ref_dq, q.grad = flash_q.grad.clone(), None
    flash_ref_dq = flash_ref_dq.transpose(1,2)

    # triton implementation
   
    a, b, c, d = q.size()
    real_q = q[:,:, rank * seq_per_rank: (rank + 1) * seq_per_rank, :].view(a, b, -1, d).contiguous().clone().detach().requires_grad_(True)
    real_k = k[:,:, rank * seq_per_rank: (rank + 1) * seq_per_rank, :].view(a, KVH, -1, d).contiguous().clone().detach().requires_grad_(True)
    real_v = v[:,:, rank * seq_per_rank: (rank + 1) * seq_per_rank, :].view(a, KVH, -1, d).contiguous().clone().detach().requires_grad_(True)
    real_do = dout[:,:, rank * seq_per_rank: (rank + 1) * seq_per_rank, :].view(a, b, -1, d).contiguous().clone().detach().requires_grad_(True)
    
    tri_out = dist_attn_varlen(real_q, real_k, real_v, causal, sm_scale).half()

    # compare
    assert torch.allclose(flash_ref_out[:, :, rank * seq_per_rank: (rank + 1) * seq_per_rank, :], tri_out, atol=1e-2, rtol=0), f" rank {rank} fails forward against flash"
    print(f" *** rank {rank} passes forward")
    tri_out.backward(real_do)
    tri_dv, real_v.grad = real_v.grad.clone(), None
    tri_dk, real_k.grad = real_k.grad.clone(), None
    tri_dq, real_q.grad = real_q.grad.clone(), None
    assert torch.allclose(flash_ref_dq[:, :, rank * seq_per_rank: (rank + 1) * seq_per_rank, :], tri_dq, atol=1e-2, rtol=0),  f" rank {rank} fails backward dq against flash"
    #print(ref_dk[:, :, rank * seq_per_rank: (rank + 1) * seq_per_rank, :].shape, ref_dk.shape, tri_dk.shape)
    assert torch.allclose(flash_ref_dk[:, :, rank * seq_per_rank: (rank + 1) * seq_per_rank, :], tri_dk, atol=1e-2, rtol=0),  f"rank {rank} fails backward dk against flash  {flash_ref_dk[:, :, rank * seq_per_rank: (rank + 1) * seq_per_rank, :]} {tri_dk} {torch.max(ref_dk[:, :, rank * seq_per_rank: (rank + 1) * seq_per_rank, :] - tri_dk)} rank {rank} fails backward dk"
    assert torch.allclose(flash_ref_dv[:, :, rank * seq_per_rank: (rank + 1) * seq_per_rank, :], tri_dv, atol=1e-2, rtol=0),  f"rank {rank} fails backward dv against flash {flash_ref_dv[:, :, rank * seq_per_rank: (rank + 1) * seq_per_rank, :]} {tri_dv} {torch.max(flash_ref_dv[:, :, rank * seq_per_rank: (rank + 1) * seq_per_rank, :] - tri_dv)} rank {rank} fails backward dv"
    print(f"rank {rank} passes backward against flash")

    assert torch.allclose(ref_out[:, :, rank * seq_per_rank: (rank + 1) * seq_per_rank, :], tri_out, atol=1e-2, rtol=0), f" rank {rank} fails forward"
    print(f" *** rank {rank} passes forward")
    assert torch.allclose(ref_dq[:, :, rank * seq_per_rank: (rank + 1) * seq_per_rank, :], tri_dq, atol=1e-2, rtol=0),  f" rank {rank} fails backward dq"
    #print(ref_dk[:, :, rank * seq_per_rank: (rank + 1) * seq_per_rank, :].shape, ref_dk.shape, tri_dk.shape)
    assert torch.allclose(ref_dk[:, :, rank * seq_per_rank: (rank + 1) * seq_per_rank, :], tri_dk, atol=1e-2, rtol=0),  f"rank {rank} fails backward dk  {ref_dk[:, :, rank * seq_per_rank: (rank + 1) * seq_per_rank, :]} {tri_dk} {torch.max(ref_dk[:, :, rank * seq_per_rank: (rank + 1) * seq_per_rank, :] - tri_dk)} rank {rank} fails backward dk"
    assert torch.allclose(ref_dv[:, :, rank * seq_per_rank: (rank + 1) * seq_per_rank, :], tri_dv, atol=1e-2, rtol=0),  f"rank {rank} fails backward dv {ref_dv[:, :, rank * seq_per_rank: (rank + 1) * seq_per_rank, :]} {tri_dv} {torch.max(ref_dv[:, :, rank * seq_per_rank: (rank + 1) * seq_per_rank, :] - tri_dv)} rank {rank} fails backward dv"
    print(f"rank {rank} passes backward")

#BATCH, N_HEADS, N_CTX, D_HEAD = 4, 48, 4096, 64
try:
    from flash_attn.flash_attn_interface import \
        flash_attn_qkvpacked_func as flash_attn_func
    FLASH_VER = 2
except BaseException:
    try:
        from flash_attn.flash_attn_interface import flash_attn_func
        FLASH_VER = 1
    except BaseException:
        FLASH_VER = None
HAS_FLASH = FLASH_VER is not None
HAS_FLASH = None
ONLY_FLASH = False

#BATCH, N_HEADS, N_CTX, D_HEAD = 4, 48, 4096, 64
BATCH, N_HEADS, N_CTX, D_HEAD = 1, 32, None, 128
# vary seq length for fixed head and batch=4
configs = [triton.testing.Benchmark(
    x_names=['N_CTX'],
    x_vals=[2**i for i in range(18, 19)],#[ 20, 21]],#[10, 11, 12, 13, 14, 15, 16, 17, 18]],
    line_arg='provider',
    line_vals=['triton'] if not ONLY_FLASH else [] + (['flash'] if HAS_FLASH else []),
    line_names=['Triton'] if not ONLY_FLASH else [] + ([f'Flash-{FLASH_VER}'] if HAS_FLASH else []),
    styles=[('red', '-'), ('blue', '-')],
    ylabel='ms',
    plot_name=f'fused-attention-batch{BATCH}-head{N_HEADS}-d{D_HEAD}-{mode}-{causal}',
    args={'H': N_HEADS, 'BATCH': BATCH, 'D_HEAD': D_HEAD, 'dtype': torch.bfloat16, 'mode': mode, 'causal': causal}
) for mode in ["all"] for causal in [True]]

# @triton.testing.perf_report(configs)
def bench_flash_attention(BATCH, H, KVH, N_CTX, D_HEAD, causal, mode, provider, args, dtype=torch.bfloat16, device="cuda"):
    assert mode == "all" #mode in ['fwd', 'bwd']
    n_warmup = 10
    n_repeat = 10
    cache = torch.empty(int(256e6), dtype=torch.int8, device='cuda')
    seq_rank = get_sequence_parallel_rank()
    seq_world_size = get_sequence_parallel_size()
    if provider == "triton":
        q = torch.randn((BATCH, H, N_CTX // seq_world_size, D_HEAD), dtype=dtype, device="cuda", requires_grad=True)
        k = torch.randn((BATCH, KVH, N_CTX // seq_world_size, D_HEAD), dtype=dtype, device="cuda", requires_grad=True)
        v = torch.randn((BATCH, KVH, N_CTX // seq_world_size, D_HEAD), dtype=dtype, device="cuda", requires_grad=True)
        if seq_rank == 0:
            print(f"Benchmarking per GPU qkv shape: {q.shape}")
        sm_scale = 1.3
        fwd_fn = lambda: dist_attn_varlen(q, k, v, causal, sm_scale)
    if provider == "flash":
        qkv = torch.randn((BATCH, N_CTX, 3, H, D_HEAD), dtype=dtype, device=device, requires_grad=True)
        if FLASH_VER == 1:
            lengths = torch.full((BATCH,), fill_value=N_CTX, device=device)
            cu_seqlens = torch.zeros((BATCH + 1,), device=device, dtype=torch.int32)
            cu_seqlens[1:] = lengths.cumsum(0)
            qkv = qkv.reshape(BATCH * N_CTX, 3, H, D_HEAD)
            fwd_fn = lambda: flash_attn_func(qkv, cu_seqlens, 0., N_CTX, causal=causal)
        elif FLASH_VER == 2:
            fwd_fn = lambda: flash_attn_func(qkv, causal=causal)
        else:
            raise ValueError(f'unknown {FLASH_VER = }')

    flops_per_matmul = 2. * BATCH * H * N_CTX * N_CTX * D_HEAD / seq_world_size
    attn_flops = 2 * flops_per_matmul
    
    assert causal
    if causal:
        attn_flops *= 0.5
    fwd_flops = attn_flops
    bwd_flops = attn_flops * 2.5 # 2.0(bwd) + 0.5(recompute)
   
    o = fwd_fn()
    do = torch.randn_like(o)
    bwd_fn = lambda: o.backward(do, retain_graph=True)

    def run_benchmark(fn):
        time_list = []
        for _ in tqdm(range(n_warmup)):
            cache.zero_()
            fn()
            torch.cuda.synchronize()
            if args.debug:
                print_and_reset_comm_stats()
        for i in tqdm(range(n_repeat)):
            cache.zero_()
            torch.cuda.synchronize()
            time_s = time.time()
            fn()
            torch.cuda.synchronize()
            time_e = time.time()
            time_list.append((time_e - time_s) * 1000.0)
            if args.debug:
                print_and_reset_comm_stats()
        return np.asarray(time_list)

    fwd_time_arr = run_benchmark(fwd_fn)
    bwd_time_arr = run_benchmark(bwd_fn)

    fwd_flops_ps = fwd_flops / np.mean(fwd_time_arr) * 1e-9
    print(f"(FWD) R={seq_rank} avg: {np.mean(fwd_time_arr)}, std: {np.std(fwd_time_arr)} flops: {fwd_flops_ps} \n")

    bwd_flops_ps = bwd_flops / np.mean(bwd_time_arr) * 1e-9
    print(f"(BWD) R={seq_rank} avg: {np.mean(bwd_time_arr)}, std: {np.std(bwd_time_arr)} flops: {bwd_flops_ps} \n")

    # total
    total_time_arr = fwd_time_arr + bwd_time_arr
    total_flops = fwd_flops + bwd_flops
    total_flops_ps = total_flops / np.mean(total_time_arr) * 1e-9
    print(f"(Total) R={seq_rank} avg: {np.mean(total_time_arr)}, std: {np.std(total_time_arr)} flops: {total_flops_ps} \n")
    
    #return total_flops_ps


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--comm-mode", type=str, default="lightseq")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--run-mode", type=str, default="test")
    parser.add_argument("--bs", type=int, default=1)
    parser.add_argument("--n_heads", type=int, default=32)
    parser.add_argument("--n_kvheads", type=int, default=32)
    parser.add_argument("--d_head", type=int, default=128)
    parser.add_argument("--start_ctx", type=int, default=12) 
    parser.add_argument("--end_ctx", type=int, default=18)
    parser.add_argument("--forward_engine", type=str, default="triton")
    parser.add_argument("--backward_engine", type=str, default="flash")

    global args
    args = parser.parse_args()
    initialize_distributed()

    assert args.forward_engine == "triton", "Only triton forward is implmented."
    assert args.backward_engine in ["flash", "xformers"], "Only flash or xformers backward is implemented."

    if args.backward_engine == "flash":
        from flash_attn.flash_attn_interface import _flash_attn_forward, _flash_attn_backward
    else:
        try:
            import xformers.ops
            from xformers.ops.fmha.common import Inputs, Context
            from xformers.ops.fmha import _memory_efficient_attention_backward
            from xformers.ops.fmha import cutlass, flash
        except ImportError:
            print("xformers not found! Please install it before trying to use it.")

    if args.run_mode == "benchmark":
        for N_CTX in [2**i for i in range(args.start_ctx, args.end_ctx)]:
            bench_flash_attention(args.bs, args.n_heads, args.n_kvheads, N_CTX, args.d_head, True, "all", "triton", args)#.run(save_path='.', print_data=True)
            reset_global_memory_buffer()
    else:
        assert args.run_mode == "test"
        for N_CTX in [4096]:
            test_op(2, 16, N_CTX, 128, True)
            #test_gqa(1, 16, 8, N_CTX, 128, True)
            reset_global_memory_buffer()
