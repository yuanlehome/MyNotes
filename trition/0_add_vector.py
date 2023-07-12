import torch
import paddle
import numpy as np

import triton
import triton.language as tl


device = "cuda:0"


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


torch.manual_seed(0)
size = 98432
x = torch.rand(size, device=device, dtype=torch.float32)
y = torch.rand(size, device=device, dtype=torch.float32)
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
        ylabel="GB/s",  # Label name for the y-axis.
        plot_name="vector-add-performance",  # Name for the plot. Used also as a file name for saving the plot.
        args={},  # Values for function arguments not in `x_names` and `y_name`.
    )
)
def benchmark(size, provider):
    x = torch.rand(size, device=device, dtype=torch.float32)
    y = torch.rand(size, device=device, dtype=torch.float32)
    x_p = paddle.to_tensor(
        data=np.random.rand(size), dtype="float32", place=paddle.CUDAPlace(0)
    )
    y_p = paddle.to_tensor(
        data=np.random.rand(size), dtype="float32", place=paddle.CUDAPlace(0)
    )
    quantiles = [0.5, 0.2, 0.8]
    if provider == "torch":
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: x + y, quantiles=quantiles
        )
    if provider == "triton":
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: add(x, y), quantiles=quantiles
        )
    if provider == "paddle":
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: x_p + y_p, quantiles=quantiles
        )
    gbps = lambda ms: 12 * size / ms * 1e-6
    return gbps(ms), gbps(max_ms), gbps(min_ms)


benchmark.run(print_data=True, show_plots=True)
# vector-add-performance:
#            size      Triton       Torch      Paddle
# 0        4096.0   13.837838   13.016949   12.000000
# 1        8192.0   26.033898   26.369099   22.925373
# 2       16384.0   46.545454   46.195491   39.384617
# 3       32768.0   69.818181   69.033707   63.999998
# 4       65536.0  117.028572  116.473930  100.721313
# 5      131072.0  154.081498  154.566035  148.495473
# 6      262144.0  192.375737  191.625729  172.766249
# 7      524288.0  209.603415  201.236440  195.241314
# 8     1048576.0  222.659124  224.310330  219.919464
# 9     2097152.0  229.883662  231.917430  231.780726
# 10    4194304.0  234.057145  238.601945  237.988205
# 11    8388608.0  236.192372  241.663062  240.848939
# 12   16777216.0  237.462725  242.351924  241.588823
# 13   33554432.0  238.312729  243.047086  243.869066
# 14   67108864.0  238.534107  243.785214  243.325546
# 15  134217728.0  239.080234  242.869983  244.347972
