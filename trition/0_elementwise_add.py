import torch
import paddle
import numpy as np

import triton
import triton.language as tl


device = "cuda:0"
dtype = torch.float32  # for benchmark


@triton.jit
def add_kernel(x_ptr, y_ptr, z_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(z_ptr + offsets, x + y, mask=mask)


def add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    z = torch.empty_like(x)
    assert x.is_cuda and y.is_cuda and z.is_cuda
    n_elements = z.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    add_kernel[grid](x, y, z, n_elements, BLOCK_SIZE=1024)
    return z


def op_test():
    torch.manual_seed(0)
    size = 98432
    x = torch.rand(size, device=device, dtype=dtype)
    y = torch.rand(size, device=device, dtype=dtype)
    z_torch = x + y
    z_triton = add(x, y)
    print(z_torch)
    print(z_triton)
    print(
        f"The maximum difference between torch and triton is "
        f"{torch.max(torch.abs(z_torch - z_triton))}"
    )


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["size"],  # Argument names to use as an x-axis for the plot.
        x_vals=[
            2**i for i in range(12, 28, 1)
        ],  # Different possible values for `x_name`.
        x_log=True,  # x axis is logarithmic.
        line_arg="provider",  # Argument name whose value corresponds to a different line in the plot.
        line_vals=["triton", "torch", "paddle"],  # Possible values for `line_arg`.
        line_names=["Triton", "Torch", "Paddle"],  # Label name for the lines.
        styles=[("blue", "-"), ("green", "-"), ("red", "-")],  # Line styles.
        ylabel="ms",  # Label name for the y-axis.
        plot_name="elementwise-add-performance",  # Name for the plot. Used also as a file name for saving the plot.
        args={},  # Values for function arguments not in `x_names` and `y_name`.
    )
)
def benchmark(size, provider):
    x = torch.rand(size, device=device, dtype=dtype)
    y = torch.rand(size, device=device, dtype=dtype)
    x_p = paddle.to_tensor(
        data=np.random.rand(size), dtype="float32", place=paddle.CUDAPlace(0)
    )
    y_p = paddle.to_tensor(
        data=np.random.rand(size), dtype="float32", place=paddle.CUDAPlace(0)
    )
    quantiles = [0.5, 0.2, 0.8]
    if provider == "torch":
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: x + y, quantiles=quantiles)
    if provider == "triton":
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: add(x, y), quantiles=quantiles
        )
    if provider == "paddle":
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: x_p + y_p, quantiles=quantiles
        )
    return ms, min_ms, max_ms


if __name__ == "__main__":
    op_test()
    benchmark.run(save_path="./perf_t4", print_data=True)
