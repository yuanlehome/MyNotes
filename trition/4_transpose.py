import torch
import paddle

import numpy as np

import triton
import triton.language as tl

device = "cuda:0"
dtype = torch.float32  # for benchmark


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


def op_test():
    torch.manual_seed(0)
    x = torch.randn(size=[1823, 781], device=device, dtype=dtype)
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
        ylabel="ms",  # label name for the y-axis
        plot_name="transpose-performance",  # name for the plot. Used also as a file name for saving the plot.
        args={"M": 4096},  # values for function arguments not in `x_names` and `y_name`
    )
)
def benchmark(M, N, provider):
    x = torch.randn(M, N, device=device, dtype=dtype)
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


if __name__ == "__main__":
    op_test()
    benchmark.run(save_path="./perf_a10", print_data=True)
