"""
Convergence tests for Liger GQA kernel.

Builds a mini transformer model with GQA attention and trains it for multiple steps,
verifying that the Liger kernel produces matching loss, logprobs, and model parameters
compared to a pure PyTorch reference implementation.
"""

import math
import os

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import pytest
import torch
import torch.nn as nn

from test.utils import assert_verbose_allclose
from test.utils import require_deterministic
from test.utils import set_seed
from test.utils import supports_bfloat16

from liger_kernel.transformers.gqa import LigerGQA
from liger_kernel.utils import infer_device

device = infer_device()


# =============================================================================
# Reference GQA module (pure PyTorch, no Triton)
# =============================================================================


class TorchGQA(nn.Module):
    """Reference GQA using repeat_interleave to expand K/V heads."""

    def __init__(
        self,
        hidden_size: int,
        num_q_heads: int,
        num_kv_heads: int,
        bias: bool = True,
        is_causal: bool = True,
    ):
        super().__init__()
        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads
        self.group_size = num_q_heads // num_kv_heads
        self.head_dim = hidden_size // num_q_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.is_causal = is_causal

        self.q_proj = nn.Linear(hidden_size, num_q_heads * self.head_dim, bias=bias)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * self.head_dim, bias=bias)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * self.head_dim, bias=bias)
        self.out_proj = nn.Linear(num_q_heads * self.head_dim, hidden_size, bias=bias)

    def forward(self, hidden_states):
        batch_size, seq_len, _ = hidden_states.shape

        q = self.q_proj(hidden_states).view(batch_size, seq_len, self.num_q_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        k = k.repeat_interleave(self.group_size, dim=1)
        v = v.repeat_interleave(self.group_size, dim=1)

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        if self.is_causal:
            causal_mask = torch.triu(
                torch.ones(seq_len, seq_len, device=scores.device, dtype=torch.bool), diagonal=1
            )
            scores = scores.masked_fill(causal_mask, float("-inf"))

        attn_weights = torch.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, v)

        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        return self.out_proj(attn_output)


# =============================================================================
# Mini Transformer block and model
# =============================================================================


class MiniTransformerBlock(nn.Module):
    """Single transformer block: LayerNorm -> Attention -> Residual -> LayerNorm -> FFN -> Residual"""

    def __init__(self, hidden_size, num_q_heads, num_kv_heads, ffn_size, attn_module):
        super().__init__()
        self.ln1 = nn.LayerNorm(hidden_size)
        self.attn = attn_module
        self.ln2 = nn.LayerNorm(hidden_size)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, ffn_size),
            nn.GELU(),
            nn.Linear(ffn_size, hidden_size),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class MiniGQAModel(nn.Module):
    """
    Minimal causal language model with GQA attention for convergence testing.

    Architecture:
        Embedding -> N x TransformerBlock(GQA) -> LayerNorm -> LM Head
    """

    def __init__(
        self,
        vocab_size,
        hidden_size,
        num_layers,
        num_q_heads,
        num_kv_heads,
        ffn_size,
        max_seq_len,
        use_liger=False,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.position_embedding = nn.Embedding(max_seq_len, hidden_size)

        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            if use_liger:
                attn = LigerGQA(
                    hidden_size=hidden_size,
                    num_q_heads=num_q_heads,
                    num_kv_heads=num_kv_heads,
                    bias=True,
                    is_causal=True,
                )
            else:
                attn = TorchGQA(
                    hidden_size=hidden_size,
                    num_q_heads=num_q_heads,
                    num_kv_heads=num_kv_heads,
                    bias=True,
                    is_causal=True,
                )
            self.layers.append(
                MiniTransformerBlock(hidden_size, num_q_heads, num_kv_heads, ffn_size, attn)
            )

        self.ln_f = nn.LayerNorm(hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(self, input_ids):
        batch_size, seq_len = input_ids.shape
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)

        x = self.embedding(input_ids) + self.position_embedding(positions)

        for layer in self.layers:
            x = layer(x)

        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits


def copy_model_weights(src_model, dst_model):
    """Copy all weights from src to dst model."""
    with torch.no_grad():
        for (src_name, src_param), (dst_name, dst_param) in zip(
            src_model.named_parameters(), dst_model.named_parameters()
        ):
            dst_param.copy_(src_param)


@require_deterministic
def run_gqa_convergence(
    num_steps=32,
    dtype=torch.bfloat16,
    lr=1e-4,
    use_liger=False,
    # Model config
    vocab_size=256,
    hidden_size=128,
    num_layers=2,
    num_q_heads=8,
    num_kv_heads=2,
    ffn_size=256,
    max_seq_len=64,
    batch_size=4,
    seq_len=32,
):
    """
    Train a mini GQA model and return loss history + final model state.

    Returns dict with:
        - loss: list of loss values per step
        - model: trained model
        - logits: final eval logits
    """
    set_seed(42)

    model = MiniGQAModel(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        ffn_size=ffn_size,
        max_seq_len=max_seq_len,
        use_liger=use_liger,
    ).to(dtype).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    # Generate deterministic synthetic data
    set_seed(123)
    all_input_ids = torch.randint(0, vocab_size, (num_steps + 1, batch_size, seq_len), device=device)

    model.train()
    loss_list = []

    for step in range(num_steps):
        input_ids = all_input_ids[step]
        # Shift targets: predict next token
        targets = torch.roll(input_ids, -1, dims=1)
        targets[:, -1] = 0  # Pad the last position

        optimizer.zero_grad()
        logits = model(input_ids)
        loss = loss_fn(logits.view(-1, vocab_size), targets.view(-1))
        loss.backward()
        optimizer.step()

        loss_list.append(loss.item())
        print(f"[{'liger' if use_liger else 'torch'}] Step {step}, Loss: {loss.item():.6f}")

    # Eval step
    model.eval()
    with torch.no_grad():
        eval_input = all_input_ids[num_steps]
        eval_logits = model(eval_input)

    return {
        "loss": loss_list,
        "model": model,
        "logits": eval_logits,
    }


# =============================================================================
# Convergence Tests
# =============================================================================


@pytest.mark.parametrize(
    "num_q_heads, num_kv_heads, dtype, loss_atol, loss_rtol, param_atol, param_rtol",
    [
        pytest.param(
            8, 2,  # GQA group_size=4
            torch.float32,
            1e-4, 1e-4,
            1e-4, 1e-4,
            id="gqa_group4_fp32",
        ),
        pytest.param(
            8, 2,  # GQA group_size=4
            torch.bfloat16,
            1e-2, 5e-2,
            1e-2, 1e-2,
            marks=pytest.mark.skipif(not supports_bfloat16(), reason="bfloat16 not supported on this GPU"),
            id="gqa_group4_bf16",
        ),
        pytest.param(
            8, 1,  # MQA
            torch.float32,
            1e-4, 1e-4,
            1e-4, 1e-4,
            id="mqa_fp32",
        ),
        pytest.param(
            8, 1,  # MQA
            torch.bfloat16,
            1e-2, 5e-2,
            1e-2, 1e-2,
            marks=pytest.mark.skipif(not supports_bfloat16(), reason="bfloat16 not supported on this GPU"),
            id="mqa_bf16",
        ),
        pytest.param(
            8, 8,  # MHA (degenerate GQA)
            torch.float32,
            1e-4, 1e-4,
            1e-4, 1e-4,
            id="mha_fp32",
        ),
        pytest.param(
            8, 8,  # MHA
            torch.bfloat16,
            1e-2, 5e-2,
            1e-2, 1e-2,
            marks=pytest.mark.skipif(not supports_bfloat16(), reason="bfloat16 not supported on this GPU"),
            id="mha_bf16",
        ),
    ],
)
def test_gqa_convergence(
    num_q_heads,
    num_kv_heads,
    dtype,
    loss_atol,
    loss_rtol,
    param_atol,
    param_rtol,
):
    """
    Verify that the Liger GQA kernel converges identically to the PyTorch reference.

    Trains a mini transformer for 32 steps and compares:
    1. Per-step training loss
    2. Final eval logits
    3. All model parameters after training
    """

    common_kwargs = dict(
        num_steps=32,
        dtype=dtype,
        lr=1e-4,
        vocab_size=256,
        hidden_size=128,
        num_layers=2,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        ffn_size=256,
        max_seq_len=64,
        batch_size=4,
        seq_len=32,
    )

    # Run reference first (important: must come first for deterministic RNG)
    expected = run_gqa_convergence(use_liger=False, **common_kwargs)
    actual = run_gqa_convergence(use_liger=True, **common_kwargs)

    # Compare per-step loss
    assert_verbose_allclose(
        torch.tensor(expected["loss"]),
        torch.tensor(actual["loss"]),
        atol=loss_atol,
        rtol=loss_rtol,
        extra_info="[Per-step training loss] ",
    )

    # Compare eval logits
    assert_verbose_allclose(
        expected["logits"],
        actual["logits"],
        atol=loss_atol,
        rtol=loss_rtol,
        extra_info="[Eval logits] ",
    )

    # Compare all model parameters
    for (expected_name, expected_param), (actual_name, actual_param) in zip(
        expected["model"].named_parameters(),
        actual["model"].named_parameters(),
    ):
        assert_verbose_allclose(
            expected_param,
            actual_param,
            atol=param_atol,
            rtol=param_rtol,
            extra_info=f"[Param: {expected_name}] ",
        )
