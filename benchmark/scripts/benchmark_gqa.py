import math

import torch
import triton

from utils import QUANTILES
from utils import SingleBenchmarkRunInput
from utils import SingleBenchmarkRunOutput
from utils import _test_memory
from utils import parse_benchmark_script_args
from utils import run_benchmarks

from liger_kernel.transformers.gqa import LigerGQA
from liger_kernel.utils import infer_device

device = infer_device()


class TorchGQA(torch.nn.Module):
    """
    Reference GQA implementation using standard PyTorch operations.

    Expands K/V heads to match Q heads via repeat_interleave, which is
    mathematically equivalent but less memory-efficient than true GQA.
    """

    def __init__(
        self,
        hidden_size: int,
        num_q_heads: int,
        num_kv_heads: int,
        bias: bool = True,
        dropout: float = 0.0,
        scale: float = None,
        is_causal: bool = True,
    ):
        super().__init__()

        if hidden_size % num_q_heads != 0:
            raise ValueError(
                f"hidden_size ({hidden_size}) must be divisible by num_q_heads ({num_q_heads})"
            )
        if num_q_heads % num_kv_heads != 0:
            raise ValueError(
                f"num_q_heads ({num_q_heads}) must be divisible by num_kv_heads ({num_kv_heads})"
            )

        self.hidden_size = hidden_size
        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads
        self.group_size = num_q_heads // num_kv_heads
        self.head_dim = hidden_size // num_q_heads
        self.scale = scale if scale is not None else 1.0 / math.sqrt(self.head_dim)
        self.is_causal = is_causal

        self.q_proj = torch.nn.Linear(hidden_size, num_q_heads * self.head_dim, bias=bias)
        self.k_proj = torch.nn.Linear(hidden_size, num_kv_heads * self.head_dim, bias=bias)
        self.v_proj = torch.nn.Linear(hidden_size, num_kv_heads * self.head_dim, bias=bias)
        self.out_proj = torch.nn.Linear(num_q_heads * self.head_dim, hidden_size, bias=bias)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape

        query = self.q_proj(hidden_states)
        key = self.k_proj(hidden_states)
        value = self.v_proj(hidden_states)

        query = query.view(batch_size, seq_len, self.num_q_heads, self.head_dim).transpose(1, 2)
        key = key.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # Expand K/V to match Q heads
        key = key.repeat_interleave(self.group_size, dim=1)
        value = value.repeat_interleave(self.group_size, dim=1)

        scores = torch.matmul(query, key.transpose(-2, -1)) * self.scale

        if self.is_causal:
            causal_mask = torch.triu(
                torch.ones(seq_len, seq_len, device=scores.device, dtype=torch.bool),
                diagonal=1,
            )
            scores = scores.masked_fill(causal_mask, float("-inf"))

        attn_weights = torch.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, value)

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.num_q_heads * self.head_dim)

        output = self.out_proj(attn_output)
        return output


def bench_speed_gqa(input: SingleBenchmarkRunInput) -> SingleBenchmarkRunOutput:
    seq_len = input.x
    provider = input.kernel_provider
    mode = input.kernel_operation_mode

    extra_benchmark_config = input.extra_benchmark_config
    batch_size = extra_benchmark_config["batch_size"]
    hidden_size = extra_benchmark_config["hidden_size"]
    num_q_heads = extra_benchmark_config["num_q_heads"]
    num_kv_heads = extra_benchmark_config["num_kv_heads"]
    bias = extra_benchmark_config["bias"]
    is_causal = extra_benchmark_config["is_causal"]
    dtype = extra_benchmark_config["dtype"]

    x_shape = (batch_size, seq_len, hidden_size)

    liger_attn = (
        LigerGQA(
            hidden_size=hidden_size,
            num_q_heads=num_q_heads,
            num_kv_heads=num_kv_heads,
            bias=bias,
            is_causal=is_causal,
        )
        .to(device)
        .to(dtype)
    )

    torch_attn = (
        TorchGQA(
            hidden_size=hidden_size,
            num_q_heads=num_q_heads,
            num_kv_heads=num_kv_heads,
            bias=bias,
            is_causal=is_causal,
        )
        .to(device)
        .to(dtype)
    )

    # Synchronize weights for fair comparison
    with torch.no_grad():
        torch_attn.q_proj.weight.copy_(liger_attn.q_proj.weight)
        torch_attn.k_proj.weight.copy_(liger_attn.k_proj.weight)
        torch_attn.v_proj.weight.copy_(liger_attn.v_proj.weight)
        torch_attn.out_proj.weight.copy_(liger_attn.out_proj.weight)

        if bias:
            torch_attn.q_proj.bias.copy_(liger_attn.q_proj.bias)
            torch_attn.k_proj.bias.copy_(liger_attn.k_proj.bias)
            torch_attn.v_proj.bias.copy_(liger_attn.v_proj.bias)
            torch_attn.out_proj.bias.copy_(liger_attn.out_proj.bias)

    x = torch.randn(x_shape, dtype=dtype, device=device)
    dy = torch.randn_like(x)
    x.requires_grad_(True)

    def fwd():
        if provider == "liger":
            return liger_attn(x)
        elif provider == "torch":
            return torch_attn(x)

    print(f"Starting Warmup for input size: {x_shape}")
    _ = fwd()
    if mode in ("backward", "full"):
        y = _
        y.backward(dy, retain_graph=True)
    print("Done Warmup")

    if mode == "forward":
        ms_50, ms_20, ms_80 = triton.testing.do_bench(fwd, grad_to_none=[x], rep=100, quantiles=QUANTILES)
    elif mode == "backward":
        y = fwd()
        ms_50, ms_20, ms_80 = triton.testing.do_bench(
            lambda: y.backward(dy, retain_graph=True),
            grad_to_none=[x],
            rep=100,
            quantiles=QUANTILES,
        )
    elif mode == "full":

        def full():
            y = fwd()
            y.backward(dy, retain_graph=True)

        ms_50, ms_20, ms_80 = triton.testing.do_bench(full, grad_to_none=[x], rep=100, quantiles=QUANTILES)

    return SingleBenchmarkRunOutput(
        y_20=ms_20,
        y_50=ms_50,
        y_80=ms_80,
    )


def bench_memory_gqa(input: SingleBenchmarkRunInput) -> SingleBenchmarkRunOutput:
    seq_len = input.x
    provider = input.kernel_provider

    extra_benchmark_config = input.extra_benchmark_config
    batch_size = extra_benchmark_config["batch_size"]
    hidden_size = extra_benchmark_config["hidden_size"]
    num_q_heads = extra_benchmark_config["num_q_heads"]
    num_kv_heads = extra_benchmark_config["num_kv_heads"]
    bias = extra_benchmark_config["bias"]
    is_causal = extra_benchmark_config["is_causal"]
    dtype = extra_benchmark_config["dtype"]

    x_shape = (batch_size, seq_len, hidden_size)

    liger_attn = (
        LigerGQA(
            hidden_size=hidden_size,
            num_q_heads=num_q_heads,
            num_kv_heads=num_kv_heads,
            bias=bias,
            is_causal=is_causal,
        )
        .to(device)
        .to(dtype)
    )

    torch_attn = (
        TorchGQA(
            hidden_size=hidden_size,
            num_q_heads=num_q_heads,
            num_kv_heads=num_kv_heads,
            bias=bias,
            is_causal=is_causal,
        )
        .to(device)
        .to(dtype)
    )

    with torch.no_grad():
        torch_attn.q_proj.weight.copy_(liger_attn.q_proj.weight)
        torch_attn.k_proj.weight.copy_(liger_attn.k_proj.weight)
        torch_attn.v_proj.weight.copy_(liger_attn.v_proj.weight)
        torch_attn.out_proj.weight.copy_(liger_attn.out_proj.weight)

        if bias:
            torch_attn.q_proj.bias.copy_(liger_attn.q_proj.bias)
            torch_attn.k_proj.bias.copy_(liger_attn.k_proj.bias)
            torch_attn.v_proj.bias.copy_(liger_attn.v_proj.bias)
            torch_attn.out_proj.bias.copy_(liger_attn.out_proj.bias)

    x = torch.randn(x_shape, dtype=dtype, device=device)
    dy = torch.randn_like(x)
    x.requires_grad_(True)

    def fwd():
        if provider == "liger":
            return liger_attn(x)
        elif provider == "torch":
            return torch_attn(x)

    def full():
        y = fwd()
        y.backward(dy, retain_graph=True)

    mem_50, mem_20, mem_80 = _test_memory(full, quantiles=QUANTILES)

    return SingleBenchmarkRunOutput(
        y_20=mem_20,
        y_50=mem_50,
        y_80=mem_80,
    )


if __name__ == "__main__":
    args = parse_benchmark_script_args()

    common_configs = {
        "kernel_name": "gqa",
        "x_name": "seq_len",
        "x_label": "sequence length",
        "x_values": [2**i for i in range(6, 13)],
        "kernel_providers": ["liger", "torch"],
        "extra_benchmark_configs": [
            # GQA group_size=4, fp32
            {
                "batch_size": 2,
                "hidden_size": 512,
                "num_q_heads": 8,
                "num_kv_heads": 2,
                "bias": True,
                "is_causal": True,
                "dtype": torch.float32,
            },
            # GQA group_size=4, larger model, fp32
            {
                "batch_size": 2,
                "hidden_size": 1024,
                "num_q_heads": 16,
                "num_kv_heads": 4,
                "bias": True,
                "is_causal": True,
                "dtype": torch.float32,
            },
            # MQA (single KV head), fp32
            {
                "batch_size": 2,
                "hidden_size": 512,
                "num_q_heads": 8,
                "num_kv_heads": 1,
                "bias": True,
                "is_causal": True,
                "dtype": torch.float32,
            },
            # MHA (all heads), fp32
            {
                "batch_size": 2,
                "hidden_size": 512,
                "num_q_heads": 8,
                "num_kv_heads": 8,
                "bias": True,
                "is_causal": True,
                "dtype": torch.float32,
            },
            # GQA group_size=4, bf16
            {
                "batch_size": 2,
                "hidden_size": 512,
                "num_q_heads": 8,
                "num_kv_heads": 2,
                "bias": True,
                "is_causal": True,
                "dtype": torch.bfloat16,
            },
            # GQA group_size=4, larger model, bf16
            {
                "batch_size": 2,
                "hidden_size": 1024,
                "num_q_heads": 16,
                "num_kv_heads": 4,
                "bias": True,
                "is_causal": True,
                "dtype": torch.bfloat16,
            },
            # MQA (single KV head), bf16
            {
                "batch_size": 2,
                "hidden_size": 512,
                "num_q_heads": 8,
                "num_kv_heads": 1,
                "bias": True,
                "is_causal": True,
                "dtype": torch.bfloat16,
            },
            # MHA (all heads), bf16
            {
                "batch_size": 2,
                "hidden_size": 512,
                "num_q_heads": 8,
                "num_kv_heads": 8,
                "bias": True,
                "is_causal": True,
                "dtype": torch.bfloat16,
            },
        ],
        "overwrite": args.overwrite,
    }

    run_benchmarks(
        bench_test_fn=bench_speed_gqa,
        kernel_operation_modes=["forward", "full", "backward"],
        metric_name="speed",
        metric_unit="ms",
        **common_configs,
    )

    run_benchmarks(
        bench_test_fn=bench_memory_gqa,
        kernel_operation_modes=["full"],
        metric_name="memory",
        metric_unit="MB",
        **common_configs,
    )
