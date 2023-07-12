import torch
import paddle

import numpy as np

import triton
import triton.language as tl

device = "cuda:0"


@triton.jit
def transpose_kernel(
    x_ptr, x_row_stride, y_ptr, y_row_stride, n_cols, BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    x_row_offsets = pid * x_row_stride + tl.arange(0, BLOCK_SIZE)
    mask = tl.arange(0, BLOCK_SIZE) < n_cols
    row = tl.load(x_ptr + x_row_offsets, mask=mask)
    y_col_offsets = pid + y_row_stride * tl.arange(0, BLOCK_SIZE)
    tl.store(y_ptr + y_col_offsets, row, mask=mask)


def transpose(x: torch.Tensor) -> torch.Tensor:
    n_rows, n_cols = x.shape
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    num_warps = 4
    if BLOCK_SIZE >= 2048:
        num_warps = 8
    if BLOCK_SIZE >= 4096:
        num_warps = 16
    y = torch.empty((n_cols, n_rows), dtype=x.dtype, device=x.device)
    grid = (n_rows,)
    transpose_kernel[grid](
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
y_trition = transpose(x)
print(
    f"The maximum difference between torch and triton is "
    f"{torch.max(torch.abs(x.t() - y_trition))}"
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
        plot_name="transpose-performance",  # name for the plot. Used also as a file name for saving the plot.
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
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: x.t(), quantiles=quantiles)
    if provider == "triton":
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: transpose(x), quantiles=quantiles
        )
    if provider == "paddle":
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: paddle.t(x_p), quantiles=quantiles
        )
    return ms, min_ms, max_ms


benchmark.run(show_plots=True, print_data=True)
# transpose-performance:
#           N     Triton     Torch    Paddle
# 0     256.0   0.073696  0.001440  0.038912
# 1     384.0   0.108320  0.001568  0.058688
# 2     512.0   0.143360  0.001344  0.076160
# 3     640.0   0.178176  0.001568  0.098528
# 4     768.0   0.215200  0.001440  0.120832
# ..      ...        ...       ...       ...
# 93  12160.0   9.994400  0.001440  2.118544
# 94  12288.0  10.015903  0.001664  2.091840
# 95  12416.0   9.617120  0.001664  2.130624
# 96  12544.0  12.100352  0.001536  2.183120
# 97  12672.0  10.422144  0.001664  2.166352

# [98 rows x 4 columns]
