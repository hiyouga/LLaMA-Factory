# Copyright 2025 the LlamaFactory team.
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
#
# Pure-Triton grouped GEMM and MoE scatter/gather kernels.
# Design adapted from VeOmni (ByteDance-Seed/VeOmni) group_gemm kernels.

"""Pure-Triton MoE kernels: grouped GEMM, scatter, and gather.

Provides four kernel types for fused MoE forward+backward without Python loops:
- group_gemm_same_nk: Variable-M grouped GEMM (forward & backward dX)
- group_gemm_same_mn: Variable-K grouped GEMM (backward dW)
- moe_scatter: Token dispatch to sorted expert buffers
- moe_gather: Token reduction from expert buffers
"""

import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Triton helper: grouped tile indexing with L2 cache-friendly swizzle
# ---------------------------------------------------------------------------


@triton.jit
def _get_pid_mn(pid, M, N, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, GROUP_SIZE: tl.constexpr):
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_SIZE * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
    return pid_m, pid_n


# ---------------------------------------------------------------------------
# group_gemm_same_nk: All experts share same N, K; variable M per expert
# Used for: forward (x @ W.T) and backward dX (grad @ W)
# ---------------------------------------------------------------------------


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 64, "BLOCK_K": 64, "GROUP": 8}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 64, "GROUP": 8}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP": 8}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP": 8}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 64, "GROUP": 8}, num_warps=8, num_stages=3),
    ],
    key=["N", "K"],
)
@triton.jit
def _group_gemm_same_nk_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    cumsum_M,
    max_M,
    G: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    TRANSPOSE_B: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP: tl.constexpr,
):
    pid_m, pid_n = _get_pid_mn(tl.program_id(0), max_M, N, BLOCK_M, BLOCK_N, GROUP)
    gid = tl.program_id(1).to(tl.int64)

    gtid_start = tl.load(cumsum_M + gid - 1, mask=gid > 0, other=0).to(tl.int64)
    gtid_end = tl.load(cumsum_M + gid).to(tl.int64)
    m_size = gtid_end - gtid_start

    if pid_m * BLOCK_M >= m_size:
        return

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # a is (total_M, K) row-major, offset by expert start
    a_base = a_ptr + gtid_start * K
    # b is (G, N, K) if TRANSPOSE_B else (G, K, N)
    b_base = b_ptr + gid * K * N
    # c is (total_M, N) row-major
    c_base = c_ptr + gtid_start * N

    if TRANSPOSE_B:
        # b layout: (G, N, K), we compute a @ b.T = a(M,K) @ b(N,K).T -> (M,N)
        a_ptrs = a_base + offs_m[:, None] * K + offs_k[None, :]
        b_ptrs = b_base + offs_n[:, None] * K + offs_k[None, :]
    else:
        # b layout: (G, K, N), we compute a @ b = a(M,K) @ b(K,N) -> (M,N)
        a_ptrs = a_base + offs_m[:, None] * K + offs_k[None, :]
        b_ptrs = b_base + offs_k[:, None] * N + offs_n[None, :]

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        k_offs = k_start + offs_k
        k_mask = k_offs < K

        a_block = tl.load(a_ptrs, mask=(offs_m[:, None] < m_size) & k_mask[None, :], other=0.0)

        if TRANSPOSE_B:
            b_block = tl.load(b_ptrs, mask=(offs_n[:, None] < N) & k_mask[None, :], other=0.0)
            acc += tl.dot(a_block, tl.trans(b_block))
        else:
            b_block = tl.load(b_ptrs, mask=k_mask[:, None] & (offs_n[None, :] < N), other=0.0)
            acc += tl.dot(a_block, b_block)

        if TRANSPOSE_B:
            a_ptrs += BLOCK_K
            b_ptrs += BLOCK_K
        else:
            a_ptrs += BLOCK_K
            b_ptrs += BLOCK_K * N

    c_ptrs = c_base + offs_m[:, None] * N + offs_n[None, :]
    c_mask = (offs_m[:, None] < m_size) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc.to(c_ptr.dtype.element_ty), mask=c_mask)


def group_gemm_same_nk(
    a: torch.Tensor,
    b: torch.Tensor,
    cumsum_M: torch.Tensor,
    max_M: int,
    transpose_b: bool = False,
) -> torch.Tensor:
    """Grouped GEMM where all groups share same N, K dimensions but variable M.

    Args:
        a: (total_M, K) input tensor, rows grouped by expert
        b: (G, N, K) if transpose_b else (G, K, N) weight tensor
        cumsum_M: (G,) cumulative token counts per expert
        max_M: maximum tokens any single expert has
        transpose_b: if True, compute a @ b.T; else compute a @ b

    Returns:
        c: (total_M, N) output tensor
    """
    if transpose_b:
        G, N, K = b.shape
    else:
        G, K, N = b.shape

    c = torch.empty((a.shape[0], N), dtype=a.dtype, device=a.device)

    _group_gemm_same_nk_kernel[
        (lambda meta: (triton.cdiv(max_M, meta["BLOCK_M"]) * triton.cdiv(N, meta["BLOCK_N"]), G))
    ](
        a_ptr=a,
        b_ptr=b,
        c_ptr=c,
        cumsum_M=cumsum_M,
        max_M=max_M,
        G=G,
        N=N,
        K=K,
        TRANSPOSE_B=transpose_b,
    )
    return c


# ---------------------------------------------------------------------------
# group_gemm_same_mn: All experts share same M, N (weight dims); variable K
# Used for: backward dW (grad.T @ input)
# ---------------------------------------------------------------------------


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32, "GROUP": 8}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP": 8}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP": 8}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP": 8}, num_warps=8, num_stages=3),
    ],
    key=["M", "N"],
)
@triton.jit
def _group_gemm_same_mn_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    cumsum_K,
    G: tl.constexpr,
    M: tl.constexpr,
    N: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP: tl.constexpr,
):
    pid_m, pid_n = _get_pid_mn(tl.program_id(0), M, N, BLOCK_M, BLOCK_N, GROUP)
    gid = tl.program_id(1).to(tl.int64)

    gtid_start = tl.load(cumsum_K + gid - 1, mask=gid > 0, other=0).to(tl.int64)
    gtid_end = tl.load(cumsum_K + gid).to(tl.int64)
    k_size = gtid_end - gtid_start

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # c is (G, M, N)
    c_base = c_ptr + gid * M * N

    if k_size == 0:
        c_ptrs = c_base + offs_m[:, None] * N + offs_n[None, :]
        c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
        tl.store(c_ptrs, tl.zeros((BLOCK_M, BLOCK_N), dtype=c_ptr.dtype.element_ty), mask=c_mask)
        return

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    offs_k = tl.arange(0, BLOCK_K)

    # a is (total_K, M), compute a.T @ b -> (M, N)
    # b is (total_K, N)
    a_base = a_ptr + gtid_start * M
    b_base = b_ptr + gtid_start * N

    for k_start in range(0, k_size, BLOCK_K):
        k_offs = k_start + offs_k
        k_mask = k_offs < k_size

        a_ptrs = a_base + k_offs[:, None] * M + offs_m[None, :]
        a_block_t = tl.trans(tl.load(a_ptrs, mask=k_mask[:, None] & (offs_m[None, :] < M), other=0.0))

        # Load b block: (BLOCK_K, BLOCK_N)
        b_ptrs = b_base + k_offs[:, None] * N + offs_n[None, :]
        b_block = tl.load(b_ptrs, mask=k_mask[:, None] & (offs_n[None, :] < N), other=0.0)

        acc += tl.dot(a_block_t, b_block)

    c_ptrs = c_base + offs_m[:, None] * N + offs_n[None, :]
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc.to(c_ptr.dtype.element_ty), mask=c_mask)


def group_gemm_same_mn(
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    cumsum_K: torch.Tensor,
) -> None:
    """Grouped GEMM where all groups produce same (M, N) output; variable K reduction.

    Computes: c[g] = a[s:e].T @ b[s:e] for each group g,
    where s, e are defined by cumsum_K boundaries.

    Args:
        a: (total_K, M) input tensor grouped by expert
        b: (total_K, N) input tensor grouped by expert
        c: (G, M, N) output tensor (pre-allocated)
        cumsum_K: (G,) cumulative token counts per expert
    """
    G, M, N = c.shape

    _group_gemm_same_mn_kernel[(lambda meta: (triton.cdiv(M, meta["BLOCK_M"]) * triton.cdiv(N, meta["BLOCK_N"]), G))](
        a_ptr=a,
        b_ptr=b,
        c_ptr=c,
        cumsum_K=cumsum_K,
        G=G,
        M=M,
        N=N,
    )


# ---------------------------------------------------------------------------
# moe_scatter: Dispatch tokens to sorted expert buffer positions
# ---------------------------------------------------------------------------


@triton.jit
def _moe_scatter_kernel(
    x_ptr,
    out_ptr,
    index_ptr,
    M,
    N: tl.constexpr,
    TOPK: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Scatter: for each token i, copy x[i] to out[index[i, k]] for k in 0..topk-1."""
    pid_m = tl.program_id(0).to(tl.int64)
    pid_n = tl.program_id(1)

    if pid_m >= M:
        return

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = offs_n < N

    # Load input row
    x_ptrs = x_ptr + pid_m * N + offs_n
    x_vals = tl.load(x_ptrs, mask=n_mask, other=0.0)

    # Store to each topk destination
    for k in tl.static_range(TOPK):
        dst_idx = tl.load(index_ptr + pid_m * TOPK + k).to(tl.int64)
        out_ptrs = out_ptr + dst_idx * N + offs_n
        tl.store(out_ptrs, x_vals, mask=n_mask)


def moe_scatter(x: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    """Scatter tokens to sorted expert buffer.

    For each token i and topk slot k, copies x[i] to output[index[i, k]].

    Args:
        x: (M, N) input hidden states
        index: (M, topk) scatter indices

    Returns:
        out: (M * topk, N) scattered output
    """
    M, N = x.shape
    topk = index.shape[1]
    out = torch.empty(M * topk, N, dtype=x.dtype, device=x.device)

    BLOCK_N = min(triton.next_power_of_2(N), 1024)
    grid = (M, triton.cdiv(N, BLOCK_N))

    _moe_scatter_kernel[grid](
        x_ptr=x,
        out_ptr=out,
        index_ptr=index,
        M=M,
        N=N,
        TOPK=topk,
        BLOCK_N=BLOCK_N,
    )
    return out


# ---------------------------------------------------------------------------
# moe_gather: Reduce expert outputs back to token positions (sum over topk)
# ---------------------------------------------------------------------------


@triton.jit
def _moe_gather_kernel(
    x_ptr,
    out_ptr,
    index_ptr,
    M,
    N: tl.constexpr,
    TOPK: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Gather: for each token i, out[i] = sum_k(x[index[i, k]]) over topk."""
    pid_m = tl.program_id(0).to(tl.int64)
    pid_n = tl.program_id(1)

    if pid_m >= M:
        return

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = offs_n < N

    acc = tl.zeros([BLOCK_N], dtype=tl.float32)

    for k in tl.static_range(TOPK):
        src_idx = tl.load(index_ptr + pid_m * TOPK + k).to(tl.int64)
        x_ptrs = x_ptr + src_idx * N + offs_n
        x_vals = tl.load(x_ptrs, mask=n_mask, other=0.0).to(tl.float32)
        acc += x_vals

    out_ptrs = out_ptr + pid_m * N + offs_n
    tl.store(out_ptrs, acc.to(out_ptr.dtype.element_ty), mask=n_mask)


def moe_gather(x: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    """Gather and reduce expert outputs back to original token positions.

    For each token i, sums x[index[i, k]] over all topk slots.

    Args:
        x: (M * topk, N) expert outputs in sorted buffer
        index: (M, topk) scatter indices (same as used in moe_scatter)

    Returns:
        out: (M, N) gathered output
    """
    M, topk = index.shape
    N = x.shape[1]
    out = torch.empty(M, N, dtype=x.dtype, device=x.device)

    BLOCK_N = min(triton.next_power_of_2(N), 1024)
    grid = (M, triton.cdiv(N, BLOCK_N))

    _moe_gather_kernel[grid](
        x_ptr=x,
        out_ptr=out,
        index_ptr=index,
        M=M,
        N=N,
        TOPK=topk,
        BLOCK_N=BLOCK_N,
    )
    return out
