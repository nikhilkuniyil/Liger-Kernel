"""
Liger Grouped Query Attention (GQA) nn.Module wrapper.

Provides a drop-in replacement for attention layers that use GQA,
such as those in Llama 2 70B, Mistral, and other modern LLMs.
"""

import math

import torch
import torch.nn as nn

from liger_kernel.transformers.functional import liger_gqa


class LigerGQA(nn.Module):
    """
    Grouped Query Attention module using Liger Triton kernels.

    GQA reduces memory bandwidth by having multiple query heads share
    key/value heads. This is more efficient than multi-head attention
    while maintaining better quality than multi-query attention.

    Args:
        hidden_size: Total hidden dimension
        num_q_heads: Number of query attention heads
        num_kv_heads: Number of key/value attention heads (must divide num_q_heads)
        head_dim: Dimension of each attention head (default: hidden_size // num_q_heads)
        bias: Whether to use bias in projections (default: True)
        dropout: Dropout probability (default: 0.0, not yet implemented)
        scale: Custom scaling factor (default: 1/sqrt(head_dim))
        is_causal: Whether to apply causal masking (default: True for autoregressive)
        backend: Attention backend to use:
            - "auto": prefer SDPA, fallback to Triton op if unavailable
            - "sdpa": force torch scaled_dot_product_attention path
            - "triton": force Liger Triton GQA op path

    Example:
        >>> gqa = LigerGQA(
        ...     hidden_size=4096,
        ...     num_q_heads=32,
        ...     num_kv_heads=8,  # 4 query heads per KV head
        ...     is_causal=True,
        ... )
        >>> hidden_states = torch.randn(2, 512, 4096)
        >>> output = gqa(hidden_states)
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
        backend: str = "auto",
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
        if backend not in {"auto", "sdpa", "triton"}:
            raise ValueError(f"backend must be one of ['auto', 'sdpa', 'triton'], got '{backend}'")
        self.backend = backend

        if dropout > 0.0:
            raise NotImplementedError("Dropout is not yet supported in LigerGQA")
        self.dropout = dropout

        # Query projection: full size
        self.q_proj = nn.Linear(hidden_size, num_q_heads * self.head_dim, bias=bias)

        # Key/Value projections: reduced size (shared across groups)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * self.head_dim, bias=bias)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * self.head_dim, bias=bias)

        # Output projection
        self.out_proj = nn.Linear(num_q_heads * self.head_dim, hidden_size, bias=bias)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Forward pass for GQA.

        Args:
            hidden_states: Input tensor [batch_size, seq_len, hidden_size]
            attention_mask: Optional attention mask (not yet implemented)

        Returns:
            output: [batch_size, seq_len, hidden_size]
        """
        if attention_mask is not None:
            raise NotImplementedError("Attention mask is not yet supported in LigerGQA")

        batch_size, seq_len, _ = hidden_states.shape

        # Project Q, K, V
        query = self.q_proj(hidden_states)
        key = self.k_proj(hidden_states)
        value = self.v_proj(hidden_states)

        # Reshape to [batch, heads, seq, head_dim]
        query = query.view(batch_size, seq_len, self.num_q_heads, self.head_dim).transpose(1, 2)
        key = key.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # Route through configured backend (auto/sdpa/triton).
        attn_output = liger_gqa(query, key, value, self.scale, self.is_causal, self.backend)

        # Reshape back to [batch, seq, hidden]
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.num_q_heads * self.head_dim)

        # Output projection
        output = self.out_proj(attn_output)

        return output


class LigerGQALayer(nn.Module):
    """
    GQA attention layer with pre-normalization (LayerNorm before attention).

    This follows the common transformer block pattern:
        output = hidden_states + GQA(LayerNorm(hidden_states))

    Args:
        hidden_size: Total hidden dimension
        num_q_heads: Number of query attention heads
        num_kv_heads: Number of key/value attention heads
        head_dim: Dimension of each attention head (default: hidden_size // num_q_heads)
        bias: Whether to use bias in projections
        dropout: Dropout probability (not yet implemented)
        layer_norm_eps: Epsilon for layer normalization
        scale: Custom scaling factor
        is_causal: Whether to apply causal masking
        backend: Attention backend ("auto", "sdpa", or "triton")

    Example:
        >>> layer = LigerGQALayer(
        ...     hidden_size=4096,
        ...     num_q_heads=32,
        ...     num_kv_heads=8,
        ... )
        >>> hidden_states = torch.randn(2, 512, 4096)
        >>> output = layer(hidden_states)
    """

    def __init__(
        self,
        hidden_size: int,
        num_q_heads: int,
        num_kv_heads: int,
        head_dim: int = None,
        bias: bool = True,
        dropout: float = 0.0,
        layer_norm_eps: float = 1e-5,
        scale: float = None,
        is_causal: bool = True,
        backend: str = "auto",
    ):
        super().__init__()

        self.attention = LigerGQA(
            hidden_size=hidden_size,
            num_q_heads=num_q_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            bias=bias,
            dropout=dropout,
            scale=scale,
            is_causal=is_causal,
            backend=backend,
        )

        self.layer_norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)

        if dropout > 0.0:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Forward pass with pre-normalization and residual connection.

        Args:
            hidden_states: [batch_size, seq_len, hidden_size]
            attention_mask: Optional attention mask (not yet implemented)

        Returns:
            output: [batch_size, seq_len, hidden_size]
        """
        # Pre-normalization
        normed_hidden_states = self.layer_norm(hidden_states)

        # Attention
        attn_output = self.attention(normed_hidden_states, attention_mask)

        # Dropout (if enabled)
        if self.dropout is not None:
            attn_output = self.dropout(attn_output)

        # Residual connection
        output = hidden_states + attn_output

        return output
