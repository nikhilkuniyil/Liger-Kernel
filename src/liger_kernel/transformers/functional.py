import inspect
import math

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F

from liger_kernel.ops import LigerCrossEntropyFunction
from liger_kernel.ops import LigerDyTFunction
from liger_kernel.ops import LigerFusedAddRMSNormFunction
from liger_kernel.ops import LigerFusedLinearCrossEntropyFunction
from liger_kernel.ops import LigerFusedLinearJSDFunction
from liger_kernel.ops import LigerFusedNeighborhoodAttentionFunction
from liger_kernel.ops import LigerGELUMulFunction
from liger_kernel.ops import LigerGQAFunction
from liger_kernel.ops import LigerGroupNormFunction
from liger_kernel.ops import LigerJSDFunction
from liger_kernel.ops import LigerKLDivLossFunction
from liger_kernel.ops import LigerLayerNormFunction
from liger_kernel.ops import LigerMultiTokenAttentionFunction
from liger_kernel.ops import LigerPolyNormFunction
from liger_kernel.ops import LigerQwen2VLMRopeFunction
from liger_kernel.ops import LigerRMSNormFunction
from liger_kernel.ops import LigerRopeFunction
from liger_kernel.ops import LigerSiLUMulFunction
from liger_kernel.ops import LigerSoftmaxFunction
from liger_kernel.ops import LigerSparsemaxFunction
from liger_kernel.ops import LigerTVDLossFunction


@dataclass
class CrossEntropyOutput:
    loss: torch.Tensor
    z_loss: Optional[torch.Tensor] = None
    token_accuracy: Optional[torch.Tensor] = None


# conform to the function signature in https://pytorch.org/docs/stable/generated/torch.nn.functional.cross_entropy.html
# `weight` and `size_average` are placeholders and not implemented yet
def liger_cross_entropy(
    input,
    target,
    weight=None,
    size_average=None,
    ignore_index: int = -100,
    reduce=None,
    reduction: str = "mean",
    label_smoothing: float = 0.0,
    lse_square_scale: float = 0.0,
    softcap: Optional[float] = None,
    return_z_loss: bool = False,
    return_token_accuracy: bool = False,
):
    loss, z_loss, token_accuracy = LigerCrossEntropyFunction.apply(
        input,
        target,
        weight,
        ignore_index,
        lse_square_scale,
        label_smoothing,
        reduction,
        softcap,
        return_z_loss,
        return_token_accuracy,
    )

    if not return_z_loss and not return_token_accuracy:
        return loss

    return CrossEntropyOutput(loss=loss, z_loss=z_loss, token_accuracy=token_accuracy)


def liger_fused_linear_cross_entropy(
    input,
    weight,
    target,
    bias=None,
    ce_weight=None,
    ignore_index: int = -100,
    lse_square_scale: float = 0.0,
    label_smoothing: float = 0.0,
    reduction: str = "mean",
    softcap: Optional[float] = None,
    return_z_loss: bool = False,
    accum_dtype=None,
    use_token_scaling: bool = False,
    return_token_accuracy: bool = False,
):
    loss, z_loss, token_accuracy = LigerFusedLinearCrossEntropyFunction.apply(
        input,
        weight,
        target,
        bias,
        ce_weight,
        ignore_index,
        lse_square_scale,
        label_smoothing,
        reduction,
        softcap,
        return_z_loss,
        accum_dtype,
        use_token_scaling,
        return_token_accuracy,
    )

    if not return_z_loss and not return_token_accuracy:
        return loss

    return CrossEntropyOutput(loss=loss, z_loss=z_loss, token_accuracy=token_accuracy)


def liger_fused_linear_jsd(
    student_input,
    student_weight,
    teacher_input,
    teacher_weight,
    shift_labels=None,
    jsd_beta: float = 0.5,
    ignore_index: int = -100,
    temperature: float = 1.0,
):
    return LigerFusedLinearJSDFunction.apply(
        student_input,
        student_weight,
        teacher_input,
        teacher_weight,
        shift_labels,
        jsd_beta,
        ignore_index,
        temperature,
    )


def liger_geglu(a, b):
    return LigerGELUMulFunction.apply(a, b)


def liger_group_norm(
    X,
    affine_scaling_weight,
    affine_shifting_bias,
    num_channels,
    num_groups,
    eps,
):
    return LigerGroupNormFunction.apply(
        X,
        affine_scaling_weight,
        affine_shifting_bias,
        num_channels,
        num_groups,
        eps,
    )


def liger_jsd(
    input,
    target,
    shift_labels=None,
    beta: float = 0.5,
    ignore_index: int = -100,
):
    return LigerJSDFunction.apply(
        input,
        target,
        shift_labels,
        beta,
        ignore_index,
    )


# conform to the function signature in https://pytorch.org/docs/stable/generated/torch.nn.functional.kl_div.html#torch.nn.functional.kl_div
# `size_average` and `mean` are being deprecated in torch API and are placeholders here
def liger_kl_div(
    input,
    target,
    size_average: bool = True,
    reduce: bool = True,
    reduction: str = "mean",
    log_target: bool = False,
    eps: float = 1e-10,
):
    # Note: the default reduction in torch is `mean`, but being `batchmean` in Liger
    return LigerKLDivLossFunction.apply(
        input,
        target,
        reduction,
        log_target,
        eps,
    )


def liger_sparsemax(
    input,
    dim: int = -1,
):
    return LigerSparsemaxFunction.apply(input, dim)


def liger_multi_token_attention(
    scores,
    weight,
    bias=None,
    stride: int = 1,
    padding: int = 0,
    dilation: int = 1,
    groups: int = 1,
    sparse: bool = False,
):
    """
    Functional interface for multi-token attention.

    Args:
        scores: Input tensor of shape (B, C_in, L, L)
        weight: Convolution weight tensor of shape (C_out, C_in // groups, K, K)
        bias: Optional bias tensor of shape (C_out,)
        stride: Stride for the convolution (default: 1)
        padding: Padding for the convolution (default: 0)
        dilation: Dilation factor for the convolution (default: 1)
        groups: Number of groups for the convolution (default: 1)
        sparse: Specifies if input tensors are expected to be sparse (default: False)
    Returns:
        Output tensor after applying multi-token attention.
    """
    return LigerMultiTokenAttentionFunction.apply(scores, weight, bias, stride, padding, dilation, groups, sparse)


def liger_fused_neighborhood_attention(
    query,
    key,
    value,
    kernel_size: int = 7,
    dilation: int = 1,
    scale: float = None,
):
    """
    Liger fused neighborhood attention.

    paper: https://arxiv.org/pdf/2504.16922

    Args:
        query: Query tensor of shape [batch_size, num_heads, seq_len, head_dim]
        key: Key tensor of shape [batch_size, num_heads, seq_len, head_dim]
        value: Value tensor of shape [batch_size, num_heads, seq_len, head_dim]
        kernel_size: Size of the neighborhood window (default: 7)
        dilation: Dilation factor for the neighborhood (default: 1)
        scale: Scaling factor for attention scores (default: rsqrt(head_dim))

    Returns:
        Output tensor of shape [batch_size, num_heads, seq_len, head_dim]
    """
    return LigerFusedNeighborhoodAttentionFunction.apply(query, key, value, kernel_size, dilation, scale)


def liger_gqa(
    query,
    key,
    value,
    scale: float = None,
    is_causal: bool = False,
    backend: str = "auto",
):
    """
    Liger Grouped Query Attention (GQA).

    GQA is an attention mechanism where multiple query heads share the same
    key/value heads, reducing memory bandwidth while maintaining quality.

    Reference: https://arxiv.org/abs/2305.13245

    Args:
        query: Query tensor of shape [batch_size, num_q_heads, seq_len, head_dim]
        key: Key tensor of shape [batch_size, num_kv_heads, seq_len, head_dim]
        value: Value tensor of shape [batch_size, num_kv_heads, seq_len, head_dim]
        scale: Scaling factor for attention scores (default: 1/sqrt(head_dim))
        is_causal: Whether to apply causal masking (default: False)
        backend: Attention backend to use:
            - "auto": prefer SDPA, fallback to Triton op if unavailable
            - "sdpa": force torch scaled_dot_product_attention path
            - "triton": force Liger Triton GQA op path

    Returns:
        Output tensor of shape [batch_size, num_q_heads, seq_len, head_dim]

    Note:
        num_q_heads must be divisible by num_kv_heads.
        The group size is num_q_heads // num_kv_heads.
    """
    if backend not in {"auto", "sdpa", "triton"}:
        raise ValueError(f"backend must be one of ['auto', 'sdpa', 'triton'], got '{backend}'")

    batch_size, num_q_heads, seq_len, head_dim = query.shape
    _, num_kv_heads, _, _ = key.shape

    if num_q_heads % num_kv_heads != 0:
        raise ValueError(
            f"num_q_heads ({num_q_heads}) must be divisible by num_kv_heads ({num_kv_heads})"
        )

    query = query.contiguous()
    key = key.contiguous()
    value = value.contiguous()

    if scale is None:
        scale = 1.0 / math.sqrt(head_dim)

    if backend == "triton":
        return LigerGQAFunction.apply(query, key, value, scale, is_causal)

    if not hasattr(F, "scaled_dot_product_attention"):
        if backend == "sdpa":
            raise RuntimeError("scaled_dot_product_attention is not available in this PyTorch version")
        return LigerGQAFunction.apply(query, key, value, scale, is_causal)

    sig = inspect.signature(F.scaled_dot_product_attention)
    supports_scale = "scale" in sig.parameters
    supports_enable_gqa = "enable_gqa" in sig.parameters

    # Older PyTorch versions may not expose `scale`; emulate it by pre-scaling Q.
    if supports_scale:
        query_for_sdpa = query
        scale_kwarg = scale
    else:
        query_for_sdpa = query * (scale * math.sqrt(head_dim))
        scale_kwarg = None

    kwargs = {
        "attn_mask": None,
        "dropout_p": 0.0,
        "is_causal": is_causal,
    }
    if supports_scale:
        kwargs["scale"] = scale_kwarg

    if num_q_heads != num_kv_heads and not supports_enable_gqa:
        group_size = num_q_heads // num_kv_heads
        key = key.repeat_interleave(group_size, dim=1)
        value = value.repeat_interleave(group_size, dim=1)
    elif supports_enable_gqa:
        kwargs["enable_gqa"] = num_q_heads != num_kv_heads

    return F.scaled_dot_product_attention(query_for_sdpa, key, value, **kwargs)


def liger_tvd(
    input,
    target,
    shift_labels=None,
    reduction: str = "mean",
    ignore_index: int = -100,
):
    return LigerTVDLossFunction.apply(
        input,
        target,
        shift_labels,
        reduction,
        ignore_index,
    )


def liger_layer_norm(X, W, B, eps):
    return LigerLayerNormFunction.apply(X, W, B, eps)


def liger_qwen2vl_mrope(q, k, cos, sin, mrope_section, unsqueeze_dim=1):
    return LigerQwen2VLMRopeFunction.apply(q, k, cos, sin, mrope_section, unsqueeze_dim)


def liger_rms_norm(X, W, eps, offset: float = 0.0, casting_mode: str = "llama", in_place: bool = True):
    return LigerRMSNormFunction.apply(X, W, eps, offset, casting_mode, in_place)


def liger_poly_norm(X, W, B, eps=1e-6, in_place=True):
    return LigerPolyNormFunction.apply(X, W, B, eps, in_place)


def liger_fused_add_rms_norm(X, R, W, eps, offset: float = 0.0, casting_mode: str = "llama", in_place: bool = True):
    return LigerFusedAddRMSNormFunction.apply(X, R, W, eps, offset, casting_mode, in_place)


def liger_rope(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    return LigerRopeFunction.apply(q, k, cos, sin, position_ids, unsqueeze_dim)


def liger_swiglu(a, b):
    return LigerSiLUMulFunction.apply(a, b)


def liger_softmax(x):
    return LigerSoftmaxFunction.apply(x)


def liger_dyt(x, alpha, gamma, beta):
    return LigerDyTFunction.apply(x, alpha, gamma, beta)
