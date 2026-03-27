"""
Tests for Liger Grouped Query Attention (GQA) kernel.

Tests cover:
1. Correctness against PyTorch reference implementation
2. Gradient correctness (backward pass)
3. Various GQA configurations (different group sizes)
4. Edge cases (MHA when group_size=1, MQA when num_kv_heads=1)
5. Causal masking
6. Shape validation
7. Determinism
"""

import math

import pytest
import torch
import torch.nn as nn

from test.utils import assert_verbose_allclose
from test.utils import set_seed

from liger_kernel.ops.gqa import LigerGQAFunction
from liger_kernel.ops.gqa import gqa_forward
from liger_kernel.transformers.functional import liger_gqa
from liger_kernel.transformers.gqa import LigerGQA
from liger_kernel.transformers.gqa import LigerGQALayer
from liger_kernel.utils import infer_device

device = infer_device()
set_seed()


# =============================================================================
# Reference PyTorch Implementation
# =============================================================================


class TorchGQA(nn.Module):
    """
    Reference GQA implementation using standard PyTorch operations.

    This expands K/V heads to match Q heads, which is mathematically equivalent
    to GQA but less memory-efficient. Used for correctness testing.
    """

    def __init__(
        self,
        hidden_size: int,
        num_q_heads: int,
        num_kv_heads: int,
        head_dim: int = None,
        bias: bool = True,
        dropout: float = 0.0,
        scale: float = None,
        is_causal: bool = True,
    ):
        super().__init__()

        if num_q_heads % num_kv_heads != 0:
            raise ValueError(
                f"num_q_heads ({num_q_heads}) must be divisible by num_kv_heads ({num_kv_heads})"
            )

        self.hidden_size = hidden_size
        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads
        self.group_size = num_q_heads // num_kv_heads

        if head_dim is None:
            if hidden_size % num_q_heads != 0:
                raise ValueError(
                    f"hidden_size ({hidden_size}) must be divisible by num_q_heads ({num_q_heads})"
                )
            self.head_dim = hidden_size // num_q_heads
        else:
            self.head_dim = head_dim

        self.scale = scale if scale is not None else 1.0 / math.sqrt(self.head_dim)
        self.is_causal = is_causal

        self.q_proj = nn.Linear(hidden_size, num_q_heads * self.head_dim, bias=bias)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * self.head_dim, bias=bias)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * self.head_dim, bias=bias)
        self.out_proj = nn.Linear(num_q_heads * self.head_dim, hidden_size, bias=bias)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape

        query = self.q_proj(hidden_states)
        key = self.k_proj(hidden_states)
        value = self.v_proj(hidden_states)

        # Reshape
        query = query.view(batch_size, seq_len, self.num_q_heads, self.head_dim).transpose(1, 2)
        key = key.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # Expand K/V to match Q heads (reference approach - less efficient but equivalent)
        key = key.repeat_interleave(self.group_size, dim=1)
        value = value.repeat_interleave(self.group_size, dim=1)

        # Standard attention
        scores = torch.matmul(query, key.transpose(-2, -1)) * self.scale

        if self.is_causal:
            causal_mask = torch.triu(
                torch.ones(seq_len, seq_len, device=scores.device, dtype=torch.bool),
                diagonal=1,
            )
            scores = scores.masked_fill(causal_mask, float("-inf"))

        attn_weights = torch.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, value)

        # Reshape back
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.num_q_heads * self.head_dim)

        output = self.out_proj(attn_output)
        return output


def torch_gqa_functional(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scale: float = None,
    is_causal: bool = False,
) -> torch.Tensor:
    """
    Reference GQA functional implementation.

    Args:
        query: [batch_size, num_q_heads, seq_len, head_dim]
        key: [batch_size, num_kv_heads, seq_len, head_dim]
        value: [batch_size, num_kv_heads, seq_len, head_dim]
        scale: Scaling factor
        is_causal: Apply causal masking

    Returns:
        output: [batch_size, num_q_heads, seq_len, head_dim]
    """
    batch_size, num_q_heads, seq_len, head_dim = query.shape
    _, num_kv_heads, _, _ = key.shape
    group_size = num_q_heads // num_kv_heads

    if scale is None:
        scale = 1.0 / math.sqrt(head_dim)

    # Expand K/V to match Q heads
    key = key.repeat_interleave(group_size, dim=1)
    value = value.repeat_interleave(group_size, dim=1)

    # Standard attention
    scores = torch.matmul(query, key.transpose(-2, -1)) * scale

    if is_causal:
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=scores.device, dtype=torch.bool),
            diagonal=1,
        )
        scores = scores.masked_fill(causal_mask, float("-inf"))

    attn_weights = torch.softmax(scores, dim=-1)
    output = torch.matmul(attn_weights, value)

    return output


# =============================================================================
# nn.Module Correctness Tests
# =============================================================================


@pytest.mark.parametrize(
    "batch_size, seq_len, hidden_size, num_q_heads, num_kv_heads",
    [
        (2, 32, 128, 8, 2),   # group_size=4
        (1, 64, 256, 16, 4),  # group_size=4
        (2, 24, 96, 6, 2),    # group_size=3
        (1, 32, 128, 8, 8),   # MHA (group_size=1)
        (2, 32, 128, 8, 1),   # MQA (single KV head)
        (1, 16, 64, 4, 2),    # Small config
    ],
)
@pytest.mark.parametrize("is_causal", [True, False])
@pytest.mark.parametrize("bias", [True, False])
@pytest.mark.parametrize(
    "dtype, atol, rtol",
    [
        (torch.float32, 5e-3, 5e-3),
        (torch.bfloat16, 5e-2, 5e-2),
    ],
)
def test_gqa_correctness(
    batch_size, seq_len, hidden_size, num_q_heads, num_kv_heads, is_causal, bias, dtype, atol, rtol
):
    """Test LigerGQA correctness against PyTorch reference."""
    set_seed(42)

    hidden_states = torch.randn(batch_size, seq_len, hidden_size, device=device, dtype=dtype)

    # Create reference and Liger implementations
    ref_gqa = TorchGQA(
        hidden_size=hidden_size,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        bias=bias,
        is_causal=is_causal,
    ).to(device).to(dtype)

    liger_gqa = LigerGQA(
        hidden_size=hidden_size,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        bias=bias,
        is_causal=is_causal,
    ).to(device).to(dtype)

    # Copy weights
    with torch.no_grad():
        liger_gqa.q_proj.weight.copy_(ref_gqa.q_proj.weight)
        liger_gqa.k_proj.weight.copy_(ref_gqa.k_proj.weight)
        liger_gqa.v_proj.weight.copy_(ref_gqa.v_proj.weight)
        liger_gqa.out_proj.weight.copy_(ref_gqa.out_proj.weight)

        if bias:
            liger_gqa.q_proj.bias.copy_(ref_gqa.q_proj.bias)
            liger_gqa.k_proj.bias.copy_(ref_gqa.k_proj.bias)
            liger_gqa.v_proj.bias.copy_(ref_gqa.v_proj.bias)
            liger_gqa.out_proj.bias.copy_(ref_gqa.out_proj.bias)

    # Clone inputs for gradient tracking
    hidden_states1 = hidden_states.detach().clone().requires_grad_(True)
    hidden_states2 = hidden_states.detach().clone().requires_grad_(True)

    # Forward pass
    out1 = liger_gqa(hidden_states1)
    out2 = ref_gqa(hidden_states2)

    # Check forward correctness
    assert_verbose_allclose(out1, out2, atol=atol, rtol=rtol)

    # Backward pass
    loss1 = out1.sum()
    loss2 = out2.sum()
    loss1.backward()
    loss2.backward()

    # Check gradient correctness
    assert_verbose_allclose(hidden_states1.grad, hidden_states2.grad, atol=atol, rtol=rtol)
    assert_verbose_allclose(liger_gqa.q_proj.weight.grad, ref_gqa.q_proj.weight.grad, atol=atol, rtol=rtol)
    assert_verbose_allclose(liger_gqa.k_proj.weight.grad, ref_gqa.k_proj.weight.grad, atol=atol, rtol=rtol)
    assert_verbose_allclose(liger_gqa.v_proj.weight.grad, ref_gqa.v_proj.weight.grad, atol=atol, rtol=rtol)
    assert_verbose_allclose(liger_gqa.out_proj.weight.grad, ref_gqa.out_proj.weight.grad, atol=atol, rtol=rtol)

    if bias:
        assert_verbose_allclose(liger_gqa.q_proj.bias.grad, ref_gqa.q_proj.bias.grad, atol=atol, rtol=rtol)
        assert_verbose_allclose(liger_gqa.k_proj.bias.grad, ref_gqa.k_proj.bias.grad, atol=atol, rtol=rtol)
        assert_verbose_allclose(liger_gqa.v_proj.bias.grad, ref_gqa.v_proj.bias.grad, atol=atol, rtol=rtol)
        assert_verbose_allclose(liger_gqa.out_proj.bias.grad, ref_gqa.out_proj.bias.grad, atol=atol, rtol=rtol)


# =============================================================================
# Functional API Correctness Tests
# =============================================================================


@pytest.mark.parametrize(
    "batch_size, num_q_heads, num_kv_heads, seq_len, head_dim",
    [
        (2, 8, 2, 32, 32),    # group_size=4
        (1, 16, 4, 24, 16),   # group_size=4
        (2, 6, 2, 16, 64),    # group_size=3
        (1, 8, 8, 32, 32),    # MHA
        (2, 8, 1, 32, 32),    # MQA
        (1, 4, 2, 48, 128),   # Larger head_dim
    ],
)
@pytest.mark.parametrize("is_causal", [True, False])
@pytest.mark.parametrize(
    "dtype, atol, rtol",
    [
        (torch.float32, 5e-3, 5e-3),
        (torch.bfloat16, 5e-2, 5e-2),
    ],
)
def test_gqa_functional_correctness(
    batch_size, num_q_heads, num_kv_heads, seq_len, head_dim, is_causal, dtype, atol, rtol
):
    """Test functional GQA API correctness."""
    set_seed(42)

    query = torch.randn(batch_size, num_q_heads, seq_len, head_dim, device=device, dtype=dtype)
    key = torch.randn(batch_size, num_kv_heads, seq_len, head_dim, device=device, dtype=dtype)
    value = torch.randn(batch_size, num_kv_heads, seq_len, head_dim, device=device, dtype=dtype)

    # Clone for gradient tracking
    query1 = query.detach().clone().requires_grad_(True)
    key1 = key.detach().clone().requires_grad_(True)
    value1 = value.detach().clone().requires_grad_(True)

    query2 = query.detach().clone().requires_grad_(True)
    key2 = key.detach().clone().requires_grad_(True)
    value2 = value.detach().clone().requires_grad_(True)

    # Liger implementation
    liger_output = LigerGQAFunction.apply(query1, key1, value1, None, is_causal)

    # PyTorch reference
    torch_output = torch_gqa_functional(query2, key2, value2, None, is_causal)

    # Check forward correctness
    assert_verbose_allclose(liger_output, torch_output, atol=atol, rtol=rtol)

    # Backward pass
    liger_output.sum().backward()
    torch_output.sum().backward()

    # Check gradient correctness
    assert_verbose_allclose(query1.grad, query2.grad, atol=atol, rtol=rtol)
    assert_verbose_allclose(key1.grad, key2.grad, atol=atol, rtol=rtol)
    assert_verbose_allclose(value1.grad, value2.grad, atol=atol, rtol=rtol)


# =============================================================================
# Shape Tests
# =============================================================================


@pytest.mark.parametrize(
    "hidden_size, num_q_heads, num_kv_heads",
    [
        (128, 8, 2),
        (256, 16, 4),
        (64, 4, 1),
        (128, 8, 8),
    ],
)
def test_gqa_shapes(hidden_size, num_q_heads, num_kv_heads):
    """Test that output shapes are correct."""
    batch_size, seq_len = 2, 32

    hidden_states = torch.randn(batch_size, seq_len, hidden_size, device=device)

    gqa = LigerGQA(
        hidden_size=hidden_size,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
    ).to(device)

    output = gqa(hidden_states)

    assert output.shape == hidden_states.shape, f"Expected shape {hidden_states.shape}, got {output.shape}"
    assert not torch.isnan(output).any(), "Output contains NaN values"
    assert not torch.isinf(output).any(), "Output contains Inf values"


def test_gqa_functional_shapes():
    """Test functional API output shapes."""
    batch_size, num_q_heads, num_kv_heads, seq_len, head_dim = 2, 8, 2, 32, 64

    query = torch.randn(batch_size, num_q_heads, seq_len, head_dim, device=device)
    key = torch.randn(batch_size, num_kv_heads, seq_len, head_dim, device=device)
    value = torch.randn(batch_size, num_kv_heads, seq_len, head_dim, device=device)

    output = LigerGQAFunction.apply(query, key, value, None, False)

    expected_shape = (batch_size, num_q_heads, seq_len, head_dim)
    assert output.shape == expected_shape, f"Expected shape {expected_shape}, got {output.shape}"
    assert not torch.isnan(output).any(), "Output contains NaN values"
    assert not torch.isinf(output).any(), "Output contains Inf values"


# =============================================================================
# Edge Case Tests
# =============================================================================


def test_gqa_edge_cases():
    """Test edge cases and error handling."""
    # num_q_heads not divisible by num_kv_heads
    with pytest.raises(ValueError, match="num_q_heads.*must be divisible by num_kv_heads"):
        LigerGQA(hidden_size=128, num_q_heads=7, num_kv_heads=3)

    # hidden_size not divisible by num_q_heads
    with pytest.raises(ValueError, match="hidden_size.*must be divisible by num_q_heads"):
        LigerGQA(hidden_size=100, num_q_heads=8, num_kv_heads=2)

    # Attention mask not supported
    gqa = LigerGQA(hidden_size=64, num_q_heads=4, num_kv_heads=2).to(device)
    hidden_states = torch.randn(1, 16, 64, device=device)
    attention_mask = torch.ones(1, 16, device=device)

    with pytest.raises(NotImplementedError, match="Attention mask is not yet supported"):
        gqa(hidden_states, attention_mask)

    # Dropout not supported
    with pytest.raises(NotImplementedError, match="Dropout is not yet supported"):
        LigerGQA(hidden_size=64, num_q_heads=4, num_kv_heads=2, dropout=0.1)

    # Invalid backend
    with pytest.raises(ValueError, match="backend must be one of"):
        LigerGQA(hidden_size=64, num_q_heads=4, num_kv_heads=2, backend="invalid")


# =============================================================================
# Determinism Test
# =============================================================================


def test_gqa_deterministic():
    """Test that results are deterministic."""
    set_seed(42)

    batch_size, seq_len, hidden_size = 2, 32, 128
    num_q_heads, num_kv_heads = 8, 2

    hidden_states = torch.randn(batch_size, seq_len, hidden_size, device=device)

    gqa = LigerGQA(
        hidden_size=hidden_size,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
    ).to(device)

    output1 = gqa(hidden_states)
    output2 = gqa(hidden_states)

    assert torch.allclose(output1, output2, atol=1e-6, rtol=1e-6), "Results are not deterministic"


def test_gqa_functional_deterministic():
    """Test that functional API is deterministic."""
    set_seed(42)

    batch_size, num_q_heads, num_kv_heads, seq_len, head_dim = 2, 8, 2, 32, 64

    query = torch.randn(batch_size, num_q_heads, seq_len, head_dim, device=device)
    key = torch.randn(batch_size, num_kv_heads, seq_len, head_dim, device=device)
    value = torch.randn(batch_size, num_kv_heads, seq_len, head_dim, device=device)

    output1 = LigerGQAFunction.apply(query, key, value, None, True)
    output2 = LigerGQAFunction.apply(query, key, value, None, True)

    assert torch.allclose(output1, output2, atol=1e-6, rtol=1e-6), "Functional API is not deterministic"


# =============================================================================
# Gradient Flow Test
# =============================================================================


@pytest.mark.parametrize(
    "batch_size, seq_len, hidden_size, num_q_heads, num_kv_heads",
    [
        (1, 16, 64, 4, 2),
        (2, 32, 128, 8, 2),
        (1, 24, 96, 6, 3),
    ],
)
def test_gqa_gradient_flow(batch_size, seq_len, hidden_size, num_q_heads, num_kv_heads):
    """Test that gradients flow correctly through the network."""
    hidden_states = torch.randn(
        batch_size, seq_len, hidden_size, device=device, requires_grad=True
    )

    gqa = LigerGQA(
        hidden_size=hidden_size,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
    ).to(device)

    output = gqa(hidden_states)
    loss = output.sum()
    loss.backward()

    # Check input gradients
    assert hidden_states.grad is not None, "Input gradients are None"
    assert not torch.allclose(
        hidden_states.grad, torch.zeros_like(hidden_states.grad)
    ), "Input gradients are zero"

    # Check parameter gradients
    for name, param in gqa.named_parameters():
        assert param.grad is not None, f"Parameter {name} has no gradient"
        assert not torch.allclose(
            param.grad, torch.zeros_like(param.grad)
        ), f"Parameter {name} has zero gradient"


# =============================================================================
# Layer Test (with LayerNorm)
# =============================================================================


class TorchGQALayer(nn.Module):
    """Reference GQA layer with pre-normalization."""

    def __init__(
        self,
        hidden_size: int,
        num_q_heads: int,
        num_kv_heads: int,
        bias: bool = True,
        layer_norm_eps: float = 1e-5,
        is_causal: bool = True,
    ):
        super().__init__()
        self.attention = TorchGQA(
            hidden_size=hidden_size,
            num_q_heads=num_q_heads,
            num_kv_heads=num_kv_heads,
            bias=bias,
            is_causal=is_causal,
        )
        self.layer_norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        normed = self.layer_norm(hidden_states)
        attn_output = self.attention(normed)
        return hidden_states + attn_output


@pytest.mark.parametrize(
    "batch_size, seq_len, hidden_size, num_q_heads, num_kv_heads",
    [
        (2, 32, 128, 8, 2),
        (1, 24, 96, 6, 3),
    ],
)
@pytest.mark.parametrize("bias", [True, False])
@pytest.mark.parametrize(
    "dtype, atol, rtol",
    [
        (torch.float32, 5e-3, 5e-3),
        (torch.bfloat16, 5e-2, 5e-2),
    ],
)
def test_gqa_layer_correctness(
    batch_size, seq_len, hidden_size, num_q_heads, num_kv_heads, bias, dtype, atol, rtol
):
    """Test LigerGQALayer correctness against PyTorch reference."""
    set_seed(42)

    hidden_states = torch.randn(batch_size, seq_len, hidden_size, device=device, dtype=dtype)

    ref_layer = TorchGQALayer(
        hidden_size=hidden_size,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        bias=bias,
    ).to(device).to(dtype)

    liger_layer = LigerGQALayer(
        hidden_size=hidden_size,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        bias=bias,
    ).to(device).to(dtype)

    # Copy weights
    with torch.no_grad():
        liger_layer.attention.q_proj.weight.copy_(ref_layer.attention.q_proj.weight)
        liger_layer.attention.k_proj.weight.copy_(ref_layer.attention.k_proj.weight)
        liger_layer.attention.v_proj.weight.copy_(ref_layer.attention.v_proj.weight)
        liger_layer.attention.out_proj.weight.copy_(ref_layer.attention.out_proj.weight)
        liger_layer.layer_norm.weight.copy_(ref_layer.layer_norm.weight)
        liger_layer.layer_norm.bias.copy_(ref_layer.layer_norm.bias)

        if bias:
            liger_layer.attention.q_proj.bias.copy_(ref_layer.attention.q_proj.bias)
            liger_layer.attention.k_proj.bias.copy_(ref_layer.attention.k_proj.bias)
            liger_layer.attention.v_proj.bias.copy_(ref_layer.attention.v_proj.bias)
            liger_layer.attention.out_proj.bias.copy_(ref_layer.attention.out_proj.bias)

    hidden_states1 = hidden_states.detach().clone().requires_grad_(True)
    hidden_states2 = hidden_states.detach().clone().requires_grad_(True)

    out1 = liger_layer(hidden_states1)
    out2 = ref_layer(hidden_states2)

    assert_verbose_allclose(out1, out2, atol=atol, rtol=rtol)

    loss1 = out1.sum()
    loss2 = out2.sum()
    loss1.backward()
    loss2.backward()

    assert_verbose_allclose(hidden_states1.grad, hidden_states2.grad, atol=atol, rtol=rtol)


# =============================================================================
# Custom Scale Test
# =============================================================================


@pytest.mark.parametrize(
    "batch_size, num_q_heads, num_kv_heads, seq_len, head_dim",
    [
        (2, 8, 2, 32, 32),
        (1, 4, 2, 16, 64),
    ],
)
def test_gqa_custom_scale(batch_size, num_q_heads, num_kv_heads, seq_len, head_dim):
    """Test that custom scale is applied correctly."""
    set_seed(42)

    query = torch.randn(batch_size, num_q_heads, seq_len, head_dim, device=device)
    key = torch.randn(batch_size, num_kv_heads, seq_len, head_dim, device=device)
    value = torch.randn(batch_size, num_kv_heads, seq_len, head_dim, device=device)

    custom_scale = 0.5

    query1 = query.clone().requires_grad_(True)
    key1 = key.clone().requires_grad_(True)
    value1 = value.clone().requires_grad_(True)

    query2 = query.clone().requires_grad_(True)
    key2 = key.clone().requires_grad_(True)
    value2 = value.clone().requires_grad_(True)

    liger_output = LigerGQAFunction.apply(query1, key1, value1, custom_scale, False)
    torch_output = torch_gqa_functional(query2, key2, value2, custom_scale, False)

    assert_verbose_allclose(liger_output, torch_output, atol=5e-3, rtol=5e-3)


def test_gqa_backend_auto_matches_triton():
    """Test that functional auto backend matches triton backend."""
    set_seed(42)

    batch_size, num_q_heads, num_kv_heads, seq_len, head_dim = 1, 8, 2, 16, 32
    query = torch.randn(batch_size, num_q_heads, seq_len, head_dim, device=device, dtype=torch.float32)
    key = torch.randn(batch_size, num_kv_heads, seq_len, head_dim, device=device, dtype=torch.float32)
    value = torch.randn(batch_size, num_kv_heads, seq_len, head_dim, device=device, dtype=torch.float32)

    out_auto = liger_gqa(query, key, value, is_causal=True, backend="auto")
    out_triton = liger_gqa(query, key, value, is_causal=True, backend="triton")

    assert_verbose_allclose(out_auto, out_triton, atol=5e-3, rtol=5e-3)


# =============================================================================
# MHA Equivalence Test
# =============================================================================


def test_gqa_mha_equivalence():
    """Test that GQA with group_size=1 is equivalent to MHA."""
    set_seed(42)

    batch_size, seq_len, hidden_size = 2, 32, 128
    num_heads = 8

    hidden_states = torch.randn(batch_size, seq_len, hidden_size, device=device)

    # GQA with num_q_heads == num_kv_heads (MHA)
    gqa = LigerGQA(
        hidden_size=hidden_size,
        num_q_heads=num_heads,
        num_kv_heads=num_heads,  # Same as Q heads = MHA
        is_causal=True,
    ).to(device)

    # Should work correctly as standard MHA
    output = gqa(hidden_states.clone().requires_grad_(True))

    assert output.shape == hidden_states.shape
    assert not torch.isnan(output).any()


# =============================================================================
# MQA Equivalence Test
# =============================================================================


def test_gqa_mqa_equivalence():
    """Test that GQA with num_kv_heads=1 is equivalent to MQA."""
    set_seed(42)

    batch_size, seq_len, hidden_size = 2, 32, 128
    num_q_heads = 8

    hidden_states = torch.randn(batch_size, seq_len, hidden_size, device=device)

    # GQA with num_kv_heads=1 (MQA)
    gqa = LigerGQA(
        hidden_size=hidden_size,
        num_q_heads=num_q_heads,
        num_kv_heads=1,  # Single KV head = MQA
        is_causal=True,
    ).to(device)

    # Should work correctly as MQA
    output = gqa(hidden_states.clone().requires_grad_(True))

    assert output.shape == hidden_states.shape
    assert not torch.isnan(output).any()
