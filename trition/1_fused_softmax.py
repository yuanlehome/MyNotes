import torch
import paddle
import numpy as np

import triton
import triton.language as tl


device = "cuda:0"


@triton.jit
def softmax_kernel(
    x_ptr, x_row_stride, y_ptr, y_row_stride, n_cols, BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    x_row_offsets = pid * x_row_stride + tl.arange(0, BLOCK_SIZE)
    mask = tl.arange(0, BLOCK_SIZE) < n_cols
    # other means padding exp(-float("inf")) = 0
    row = tl.load(x_ptr + x_row_offsets, mask=mask, other=-float("inf"))
    row_minus_max = row - tl.max(row, axis=0)
    numerator = tl.exp(row_minus_max)
    denominator = tl.sum(numerator, axis=0)
    softmax_out = numerator / denominator
    y_row_offsets = pid * y_row_stride + tl.arange(0, BLOCK_SIZE)
    tl.store(y_ptr + y_row_offsets, softmax_out, mask=mask)


def softmax(x: torch.Tensor) -> torch.Tensor:
    n_rows, n_cols = x.shape
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    num_warps = 4
    if BLOCK_SIZE >= 2048:
        num_warps = 8
    if BLOCK_SIZE >= 4096:
        num_warps = 16
    y = torch.empty_like(x)
    grid = (n_rows,)
    softmax_kernel[grid](
        x,
        x.stride(0),
        y,
        y.stride(0),
        n_cols,
        num_warps=num_warps,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return y


torch.manual_seed(0)
x = torch.randn(size=[1823, 781], device=device, dtype=torch.float32)
y_torch = torch.softmax(x, axis=1)
y_triton = softmax(x)
print(
    f"The maximum difference between torch and triton is "
    f"{torch.max(torch.abs(y_torch - y_triton))}"
)


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["N"],  # argument names to use as an x-axis for the plot
        x_vals=[
            128 * i for i in range(2, 100)
        ],  # different possible values for `x_name`
        line_arg="provider",  # argument name whose value corresponds to a different line in the plot
        line_vals=[
            "triton",
            "torch",
            "paddle",
        ],  # possible values for `line_arg``
        line_names=[
            "Triton",
            "Torch",
            "Paddle",
        ],  # label name for the lines
        styles=[("blue", "-"), ("green", "-"), ("red", "-")],  # line styles
        ylabel="GB/s",  # label name for the y-axis
        plot_name="softmax-performance",  # name for the plot. Used also as a file name for saving the plot.
        args={"M": 4096},  # values for function arguments not in `x_names` and `y_name`
    )
)
def benchmark(M, N, provider):
    x = torch.randn(M, N, device=device, dtype=torch.float32)
    x_p = paddle.to_tensor(
        np.random.randn(M, N), dtype="float32", place=paddle.CUDAPlace(0)
    )
    quantiles = [0.5, 0.2, 0.8]
    if provider == "torch":
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: torch.softmax(x, axis=-1), quantiles=quantiles
        )
    if provider == "triton":
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: softmax(x), quantiles=quantiles
        )
    if provider == "paddle":
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: paddle.nn.functional.softmax(x_p, axis=-1), quantiles=quantiles
        )
    gbps = lambda ms: 2 * x.nelement() * x.element_size() * 1e-9 / (ms * 1e-3)
    return gbps(ms), gbps(max_ms), gbps(min_ms)


benchmark.run(show_plots=True, print_data=True)
# softmax-performance:
#           N      Triton       Torch      Paddle
# 0     256.0  223.101280  218.544384  215.845195
# 1     384.0  228.746939  219.551087  218.818037
# 2     512.0  226.327644  220.474353  215.578957
# 3     640.0  226.493863  214.204943  215.295671
# 4     768.0  227.555555  214.930860  215.578943
# ..      ...         ...         ...         ...
# 93  12160.0  228.845737  133.231044  225.446120
# 94  12288.0  228.692903  133.315451  225.603303
# 95  12416.0  228.714035  135.284652  225.927964
# 96  12544.0  229.105986  133.617549  225.763775
# 97  12672.0  228.601868  131.806922  225.824956

# [98 rows x 4 columns]
