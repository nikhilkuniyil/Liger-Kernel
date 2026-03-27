"""
Grouped Query Attention (GQA) Triton Kernel — FlashAttention-2 style.

Fused implementation that never materializes the full [B, H, S, S] attention matrix.
Memory: O(B * H_q * S * D + B * H_q * S) instead of O(B * H_q * S^2).

References:
- GQA: https://arxiv.org/abs/2305.13245
- FlashAttention-2: https://arxiv.org/abs/2307.08691
"""

import math

import torch
import triton
import triton.language as tl

from liger_kernel.ops.utils import ensure_contiguous


# ───────────────────────────── Forward ──────────────────────────────


@triton.jit
def _gqa_fwd_kernel(
    Q,
    K,
    V,
    O,
    LSE,
    stride_qb,
    stride_qh,
    stride_qs,
    stride_qd,
    stride_kb,
    stride_kh,
    stride_ks,
    stride_kd,
    stride_vb,
    stride_vh,
    stride_vs,
    stride_vd,
    stride_ob,
    stride_oh,
    stride_os,
    stride_od,
    stride_lseb,
    stride_lseh,
    stride_lses,
    num_q_heads: tl.constexpr,
    num_kv_heads: tl.constexpr,
    seq_len: tl.constexpr,
    head_dim: tl.constexpr,
    scale: tl.constexpr,
    is_causal: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Fused GQA forward: softmax(Q @ K^T * scale) @ V without materializing attention."""
    pid_bh = tl.program_id(0)
    pid_m = tl.program_id(1)

    batch_id = pid_bh // num_q_heads
    q_head_id = pid_bh % num_q_heads
    kv_head_id = q_head_id // (num_q_heads // num_kv_heads)

    m_offs = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    d_offs = tl.arange(0, BLOCK_D)

    # Load Q block [BLOCK_M, BLOCK_D]
    q_ptrs = (
        Q
        + batch_id * stride_qb
        + q_head_id * stride_qh
        + m_offs[:, None] * stride_qs
        + d_offs[None, :] * stride_qd
    )
    q = tl.load(
        q_ptrs,
        mask=(m_offs[:, None] < seq_len) & (d_offs[None, :] < head_dim),
        other=0.0,
    )

    # Online softmax accumulators
    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    o_i = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)

    k_base = K + batch_id * stride_kb + kv_head_id * stride_kh
    v_base = V + batch_id * stride_vb + kv_head_id * stride_vh

    for n_start in range(0, seq_len, BLOCK_N):
        n_offs = n_start + tl.arange(0, BLOCK_N)

        # Load K [BLOCK_N, BLOCK_D]
        k = tl.load(
            k_base + n_offs[:, None] * stride_ks + d_offs[None, :] * stride_kd,
            mask=(n_offs[:, None] < seq_len) & (d_offs[None, :] < head_dim),
            other=0.0,
        )

        # S = Q @ K^T * scale  [BLOCK_M, BLOCK_N]
        s = tl.dot(q, tl.trans(k)) * scale

        # Boundary and causal masks
        s = tl.where(n_offs[None, :] < seq_len, s, float("-inf"))
        if is_causal:
            s = tl.where(m_offs[:, None] >= n_offs[None, :], s, float("-inf"))

        # Online softmax update
        m_ij = tl.max(s, axis=1)
        m_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(s - m_new[:, None])
        l_new = alpha * l_i + tl.sum(p, axis=1)
        o_i = alpha[:, None] * o_i

        # Load V [BLOCK_N, BLOCK_D] and accumulate output
        v = tl.load(
            v_base + n_offs[:, None] * stride_vs + d_offs[None, :] * stride_vd,
            mask=(n_offs[:, None] < seq_len) & (d_offs[None, :] < head_dim),
            other=0.0,
        )
        o_i += tl.dot(p.to(v.dtype), v)

        m_i = m_new
        l_i = l_new

    # Final normalization
    o_i = o_i / l_i[:, None]

    # Store output
    o_ptrs = (
        O
        + batch_id * stride_ob
        + q_head_id * stride_oh
        + m_offs[:, None] * stride_os
        + d_offs[None, :] * stride_od
    )
    tl.store(
        o_ptrs,
        o_i.to(q.dtype),
        mask=(m_offs[:, None] < seq_len) & (d_offs[None, :] < head_dim),
    )

    # Store log-sum-exp for backward
    lse_ptrs = (
        LSE + batch_id * stride_lseb + q_head_id * stride_lseh + m_offs * stride_lses
    )
    tl.store(lse_ptrs, m_i + tl.log(l_i), mask=m_offs < seq_len)


# ───────────────────────────── Backward dQ ──────────────────────────


@triton.jit
def _gqa_bwd_dq_kernel(
    Q,
    K,
    V,
    dO,
    dQ,
    LSE,
    Delta,
    stride_qb,
    stride_qh,
    stride_qs,
    stride_qd,
    stride_kb,
    stride_kh,
    stride_ks,
    stride_kd,
    stride_vb,
    stride_vh,
    stride_vs,
    stride_vd,
    stride_dob,
    stride_doh,
    stride_dos,
    stride_dod,
    stride_dqb,
    stride_dqh,
    stride_dqs,
    stride_dqd,
    stride_lseb,
    stride_lseh,
    stride_lses,
    stride_deltab,
    stride_deltah,
    stride_deltas,
    num_q_heads: tl.constexpr,
    num_kv_heads: tl.constexpr,
    seq_len: tl.constexpr,
    head_dim: tl.constexpr,
    scale: tl.constexpr,
    is_causal: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Compute dQ by recomputing attention on-the-fly and iterating over KV blocks."""
    pid_bh = tl.program_id(0)
    pid_m = tl.program_id(1)

    batch_id = pid_bh // num_q_heads
    q_head_id = pid_bh % num_q_heads
    kv_head_id = q_head_id // (num_q_heads // num_kv_heads)

    m_offs = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    d_offs = tl.arange(0, BLOCK_D)
    md_mask = (m_offs[:, None] < seq_len) & (d_offs[None, :] < head_dim)

    # Load Q block
    q_base = Q + batch_id * stride_qb + q_head_id * stride_qh
    q = tl.load(
        q_base + m_offs[:, None] * stride_qs + d_offs[None, :] * stride_qd,
        mask=md_mask,
        other=0.0,
    )

    # Load dO block
    do_base = dO + batch_id * stride_dob + q_head_id * stride_doh
    do = tl.load(
        do_base + m_offs[:, None] * stride_dos + d_offs[None, :] * stride_dod,
        mask=md_mask,
        other=0.0,
    )

    # Load LSE and Delta for this Q block
    lse = tl.load(
        LSE + batch_id * stride_lseb + q_head_id * stride_lseh + m_offs * stride_lses,
        mask=m_offs < seq_len,
        other=0.0,
    )
    di = tl.load(
        Delta
        + batch_id * stride_deltab
        + q_head_id * stride_deltah
        + m_offs * stride_deltas,
        mask=m_offs < seq_len,
        other=0.0,
    )

    dq_acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)

    k_base = K + batch_id * stride_kb + kv_head_id * stride_kh
    v_base = V + batch_id * stride_vb + kv_head_id * stride_vh

    for n_start in range(0, seq_len, BLOCK_N):
        n_offs = n_start + tl.arange(0, BLOCK_N)
        nd_mask = (n_offs[:, None] < seq_len) & (d_offs[None, :] < head_dim)

        # Load K, V
        k = tl.load(
            k_base + n_offs[:, None] * stride_ks + d_offs[None, :] * stride_kd,
            mask=nd_mask,
            other=0.0,
        )
        v = tl.load(
            v_base + n_offs[:, None] * stride_vs + d_offs[None, :] * stride_vd,
            mask=nd_mask,
            other=0.0,
        )

        # Recompute attention weights
        s = tl.dot(q, tl.trans(k)) * scale
        s = tl.where(n_offs[None, :] < seq_len, s, float("-inf"))
        if is_causal:
            s = tl.where(m_offs[:, None] >= n_offs[None, :], s, float("-inf"))
        p = tl.exp(s - lse[:, None])

        # dP = dO @ V^T,  dS = P * (dP - D)
        dp = tl.dot(do, tl.trans(v))
        ds = p * (dp - di[:, None])

        # dQ += dS @ K
        dq_acc += tl.dot(ds.to(k.dtype), k)

    dq_acc *= scale

    # Store dQ
    dq_base = dQ + batch_id * stride_dqb + q_head_id * stride_dqh
    tl.store(
        dq_base + m_offs[:, None] * stride_dqs + d_offs[None, :] * stride_dqd,
        dq_acc.to(q.dtype),
        mask=md_mask,
    )


# ───────────────────────────── Backward dK, dV ─────────────────────


@triton.jit
def _gqa_bwd_dkv_kernel(
    Q,
    K,
    V,
    dO,
    dK,
    dV,
    LSE,
    Delta,
    stride_qb,
    stride_qh,
    stride_qs,
    stride_qd,
    stride_kb,
    stride_kh,
    stride_ks,
    stride_kd,
    stride_vb,
    stride_vh,
    stride_vs,
    stride_vd,
    stride_dob,
    stride_doh,
    stride_dos,
    stride_dod,
    stride_dkb,
    stride_dkh,
    stride_dks,
    stride_dkd,
    stride_dvb,
    stride_dvh,
    stride_dvs,
    stride_dvd,
    stride_lseb,
    stride_lseh,
    stride_lses,
    stride_deltab,
    stride_deltah,
    stride_deltas,
    num_q_heads: tl.constexpr,
    num_kv_heads: tl.constexpr,
    seq_len: tl.constexpr,
    head_dim: tl.constexpr,
    scale: tl.constexpr,
    is_causal: tl.constexpr,
    group_size: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Compute dK, dV by iterating over all Q heads in the group and Q blocks."""
    pid_bkv = tl.program_id(0)
    pid_n = tl.program_id(1)

    batch_id = pid_bkv // num_kv_heads
    kv_head_id = pid_bkv % num_kv_heads

    n_offs = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    d_offs = tl.arange(0, BLOCK_D)
    nd_mask = (n_offs[:, None] < seq_len) & (d_offs[None, :] < head_dim)

    # Load K, V for this KV block (shared across Q heads in group)
    k_base = K + batch_id * stride_kb + kv_head_id * stride_kh
    v_base = V + batch_id * stride_vb + kv_head_id * stride_vh

    k = tl.load(
        k_base + n_offs[:, None] * stride_ks + d_offs[None, :] * stride_kd,
        mask=nd_mask,
        other=0.0,
    )
    v = tl.load(
        v_base + n_offs[:, None] * stride_vs + d_offs[None, :] * stride_vd,
        mask=nd_mask,
        other=0.0,
    )

    dk_acc = tl.zeros([BLOCK_N, BLOCK_D], dtype=tl.float32)
    dv_acc = tl.zeros([BLOCK_N, BLOCK_D], dtype=tl.float32)

    # Accumulate over all Q heads that share this KV head
    for g in range(group_size):
        q_head_id = kv_head_id * group_size + g

        q_base = Q + batch_id * stride_qb + q_head_id * stride_qh
        do_base = dO + batch_id * stride_dob + q_head_id * stride_doh
        lse_base = LSE + batch_id * stride_lseb + q_head_id * stride_lseh
        delta_base = Delta + batch_id * stride_deltab + q_head_id * stride_deltah

        for m_start in range(0, seq_len, BLOCK_M):
            m_offs = m_start + tl.arange(0, BLOCK_M)
            md_mask_q = (m_offs[:, None] < seq_len) & (d_offs[None, :] < head_dim)

            # Load Q, dO blocks for this Q head
            q = tl.load(
                q_base + m_offs[:, None] * stride_qs + d_offs[None, :] * stride_qd,
                mask=md_mask_q,
                other=0.0,
            )
            do = tl.load(
                do_base + m_offs[:, None] * stride_dos + d_offs[None, :] * stride_dod,
                mask=md_mask_q,
                other=0.0,
            )
            lse = tl.load(
                lse_base + m_offs * stride_lses, mask=m_offs < seq_len, other=0.0
            )
            di = tl.load(
                delta_base + m_offs * stride_deltas,
                mask=m_offs < seq_len,
                other=0.0,
            )

            # Recompute attention: P = exp(Q @ K^T * scale - LSE)
            s = tl.dot(q, tl.trans(k)) * scale
            s = tl.where(n_offs[None, :] < seq_len, s, float("-inf"))
            if is_causal:
                s = tl.where(m_offs[:, None] >= n_offs[None, :], s, float("-inf"))
            p = tl.exp(s - lse[:, None])

            # dV += P^T @ dO  (keep p in fp32 to avoid bf16 precision loss)
            dv_acc += tl.dot(tl.trans(p), do.to(tl.float32))

            # dP = dO @ V^T,  dS = P * (dP - D)
            dp = tl.dot(do, tl.trans(v))
            ds = p * (dp - di[:, None])

            # dK += dS^T @ Q
            dk_acc += tl.dot(tl.trans(ds.to(q.dtype)), q)

    dk_acc *= scale

    # Store dK, dV
    dk_base = dK + batch_id * stride_dkb + kv_head_id * stride_dkh
    dv_base = dV + batch_id * stride_dvb + kv_head_id * stride_dvh

    tl.store(
        dk_base + n_offs[:, None] * stride_dks + d_offs[None, :] * stride_dkd,
        dk_acc.to(k.dtype),
        mask=nd_mask,
    )
    tl.store(
        dv_base + n_offs[:, None] * stride_dvs + d_offs[None, :] * stride_dvd,
        dv_acc.to(v.dtype),
        mask=nd_mask,
    )


# ───────────────────────────── Python wrappers ──────────────────────


def gqa_forward(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scale: float = None,
    is_causal: bool = False,
) -> tuple:
    """
    Fused GQA forward pass.

    Args:
        query: [batch_size, num_q_heads, seq_len, head_dim]
        key: [batch_size, num_kv_heads, seq_len, head_dim]
        value: [batch_size, num_kv_heads, seq_len, head_dim]
        scale: Scaling factor (default: 1/sqrt(head_dim))
        is_causal: Whether to apply causal masking

    Returns:
        (output, lse) where lse is log-sum-exp for backward recomputation
    """
    batch_size, num_q_heads, seq_len, head_dim = query.shape
    _, num_kv_heads, _, _ = key.shape

    assert num_q_heads % num_kv_heads == 0, (
        f"num_q_heads ({num_q_heads}) must be divisible by num_kv_heads ({num_kv_heads})"
    )

    if scale is None:
        scale = 1.0 / math.sqrt(head_dim)

    query = query.contiguous()
    key = key.contiguous()
    value = value.contiguous()

    output = torch.empty_like(query)
    lse = torch.empty(
        batch_size, num_q_heads, seq_len, device=query.device, dtype=torch.float32
    )

    max_block = 64 if head_dim <= 64 else 32
    BLOCK_M = max(16, min(max_block, triton.next_power_of_2(seq_len)))
    BLOCK_N = max(16, min(max_block, triton.next_power_of_2(seq_len)))
    BLOCK_D = max(16, triton.next_power_of_2(head_dim))
    num_warps = 4

    grid = (batch_size * num_q_heads, triton.cdiv(seq_len, BLOCK_M))

    _gqa_fwd_kernel[grid](
        query,
        key,
        value,
        output,
        lse,
        query.stride(0),
        query.stride(1),
        query.stride(2),
        query.stride(3),
        key.stride(0),
        key.stride(1),
        key.stride(2),
        key.stride(3),
        value.stride(0),
        value.stride(1),
        value.stride(2),
        value.stride(3),
        output.stride(0),
        output.stride(1),
        output.stride(2),
        output.stride(3),
        lse.stride(0),
        lse.stride(1),
        lse.stride(2),
        num_q_heads,
        num_kv_heads,
        seq_len,
        head_dim,
        scale,
        is_causal,
        BLOCK_M,
        BLOCK_N,
        BLOCK_D,
        num_warps=num_warps,
        num_stages=2,
    )

    return output, lse


def gqa_backward(
    grad_output: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    output: torch.Tensor,
    lse: torch.Tensor,
    scale: float,
    is_causal: bool,
) -> tuple:
    """
    Fused GQA backward pass — recomputes attention on-the-fly.

    Args:
        grad_output: [batch_size, num_q_heads, seq_len, head_dim]
        query, key, value: Saved tensors from forward
        output: Forward output (for Delta precomputation)
        lse: Log-sum-exp from forward [batch_size, num_q_heads, seq_len]
        scale: Scale factor used in forward
        is_causal: Whether causal masking was used

    Returns:
        (grad_query, grad_key, grad_value)
    """
    batch_size, num_q_heads, seq_len, head_dim = query.shape
    _, num_kv_heads, _, _ = key.shape
    group_size = num_q_heads // num_kv_heads

    grad_output = grad_output.contiguous()

    # Precompute Delta_i = rowsum(dO_i * O_i)  [B, H_q, S]
    delta = (grad_output * output).sum(dim=-1).contiguous()

    grad_query = torch.empty_like(query)
    grad_key = torch.empty_like(key)
    grad_value = torch.empty_like(value)

    max_block = 64 if head_dim <= 64 else 32
    BLOCK_M = max(16, min(max_block, triton.next_power_of_2(seq_len)))
    BLOCK_N = max(16, min(max_block, triton.next_power_of_2(seq_len)))
    BLOCK_D = max(16, triton.next_power_of_2(head_dim))
    num_warps = 4

    # dQ kernel: grid over (batch * q_heads, q_blocks)
    grid_dq = (batch_size * num_q_heads, triton.cdiv(seq_len, BLOCK_M))
    _gqa_bwd_dq_kernel[grid_dq](
        query,
        key,
        value,
        grad_output,
        grad_query,
        lse,
        delta,
        query.stride(0),
        query.stride(1),
        query.stride(2),
        query.stride(3),
        key.stride(0),
        key.stride(1),
        key.stride(2),
        key.stride(3),
        value.stride(0),
        value.stride(1),
        value.stride(2),
        value.stride(3),
        grad_output.stride(0),
        grad_output.stride(1),
        grad_output.stride(2),
        grad_output.stride(3),
        grad_query.stride(0),
        grad_query.stride(1),
        grad_query.stride(2),
        grad_query.stride(3),
        lse.stride(0),
        lse.stride(1),
        lse.stride(2),
        delta.stride(0),
        delta.stride(1),
        delta.stride(2),
        num_q_heads,
        num_kv_heads,
        seq_len,
        head_dim,
        scale,
        is_causal,
        BLOCK_M,
        BLOCK_N,
        BLOCK_D,
        num_warps=num_warps,
        num_stages=2,
    )

    # dK, dV kernel: grid over (batch * kv_heads, kv_blocks)
    grid_dkv = (batch_size * num_kv_heads, triton.cdiv(seq_len, BLOCK_N))
    _gqa_bwd_dkv_kernel[grid_dkv](
        query,
        key,
        value,
        grad_output,
        grad_key,
        grad_value,
        lse,
        delta,
        query.stride(0),
        query.stride(1),
        query.stride(2),
        query.stride(3),
        key.stride(0),
        key.stride(1),
        key.stride(2),
        key.stride(3),
        value.stride(0),
        value.stride(1),
        value.stride(2),
        value.stride(3),
        grad_output.stride(0),
        grad_output.stride(1),
        grad_output.stride(2),
        grad_output.stride(3),
        grad_key.stride(0),
        grad_key.stride(1),
        grad_key.stride(2),
        grad_key.stride(3),
        grad_value.stride(0),
        grad_value.stride(1),
        grad_value.stride(2),
        grad_value.stride(3),
        lse.stride(0),
        lse.stride(1),
        lse.stride(2),
        delta.stride(0),
        delta.stride(1),
        delta.stride(2),
        num_q_heads,
        num_kv_heads,
        seq_len,
        head_dim,
        scale,
        is_causal,
        group_size,
        BLOCK_M,
        BLOCK_N,
        BLOCK_D,
        num_warps=num_warps,
        num_stages=2,
    )

    return grad_query, grad_key, grad_value


# ───────────────────────────── Autograd ─────────────────────────────


class LigerGQAFunction(torch.autograd.Function):
    """PyTorch autograd Function for fused Grouped Query Attention."""

    @staticmethod
    @ensure_contiguous
    def forward(
        ctx,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        scale: float = None,
        is_causal: bool = False,
    ):
        """
        Forward pass for GQA.

        Args:
            query: [batch_size, num_q_heads, seq_len, head_dim]
            key: [batch_size, num_kv_heads, seq_len, head_dim]
            value: [batch_size, num_kv_heads, seq_len, head_dim]
            scale: Scaling factor (default: 1/sqrt(head_dim))
            is_causal: Apply causal masking

        Returns:
            output: [batch_size, num_q_heads, seq_len, head_dim]
        """
        head_dim = query.shape[-1]
        if scale is None:
            scale = 1.0 / math.sqrt(head_dim)

        output, lse = gqa_forward(query, key, value, scale, is_causal)

        ctx.save_for_backward(query, key, value, output, lse)
        ctx.scale = scale
        ctx.is_causal = is_causal

        return output

    @staticmethod
    @ensure_contiguous
    def backward(ctx, grad_output: torch.Tensor):
        """
        Backward pass for GQA.

        Returns:
            Gradients for (query, key, value, None, None)
        """
        query, key, value, output, lse = ctx.saved_tensors

        grad_query, grad_key, grad_value = gqa_backward(
            grad_output,
            query,
            key,
            value,
            output,
            lse,
            ctx.scale,
            ctx.is_causal,
        )

        return grad_query, grad_key, grad_value, None, None
