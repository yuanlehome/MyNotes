import torch
import paddle
import numpy as np

import triton
import triton.language as tl


device = "cuda:0"
dtype = torch.float32  # for benchmark


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


def op_test():
    torch.manual_seed(0)
    x = torch.randn(size=[1823, 781], device=device, dtype=dtype)
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
        ylabel="ms",  # label name for the y-axis
        plot_name="softmax-performance",  # name for the plot. Used also as a file name for saving the plot.
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
    return ms, min_ms, max_ms


if __name__ == "__main__":
    op_test()
    benchmark.run(save_path="./perperf_a10_cuda11.8_cudnn8.6f_a10", print_data=True)
