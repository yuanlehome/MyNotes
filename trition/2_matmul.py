import torch
import paddle

import numpy as np

import triton
import triton.language as tl


device = "cuda:0"
dtype = torch.float16  # for benchmark


@triton.autotune(
    configs=[
        triton.Config(
            {
                "BLOCK_SIZE_M": 128,
                "BLOCK_SIZE_N": 256,
                "BLOCK_SIZE_K": 64,
                "GROUP_SIZE_M": 8,
            },
            num_stages=3,
            num_warps=8,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_M": 64,
                "BLOCK_SIZE_N": 256,
                "BLOCK_SIZE_K": 32,
                "GROUP_SIZE_M": 8,
            },
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_M": 128,
                "BLOCK_SIZE_N": 128,
                "BLOCK_SIZE_K": 32,
                "GROUP_SIZE_M": 8,
            },
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_M": 128,
                "BLOCK_SIZE_N": 64,
                "BLOCK_SIZE_K": 32,
                "GROUP_SIZE_M": 8,
            },
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_M": 64,
                "BLOCK_SIZE_N": 128,
                "BLOCK_SIZE_K": 32,
                "GROUP_SIZE_M": 8,
            },
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_M": 128,
                "BLOCK_SIZE_N": 32,
                "BLOCK_SIZE_K": 32,
                "GROUP_SIZE_M": 8,
            },
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_M": 64,
                "BLOCK_SIZE_N": 32,
                "BLOCK_SIZE_K": 32,
                "GROUP_SIZE_M": 8,
            },
            num_stages=5,
            num_warps=2,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_M": 32,
                "BLOCK_SIZE_N": 64,
                "BLOCK_SIZE_K": 32,
                "GROUP_SIZE_M": 8,
            },
            num_stages=5,
            num_warps=2,
        ),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def matmul_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    pid = tl.program_id(0)
    grid_m = tl.cdiv(M, BLOCK_SIZE_M)
    grid_n = tl.cdiv(N, BLOCK_SIZE_N)

    # I don't understand the following code temporarily!
    num_pid_in_group = GROUP_SIZE_M * grid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(grid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offsets_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offsets_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offsets_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (
        offsets_am[:, None] * stride_am + offsets_k[None, :] * stride_ak
    )  # BLOCK_SIZE_Mx1 + 1xBLOCK_SIZE_K broadcast to BLOCK_SIZE_MxBLOCK_SIZE_K
    b_ptrs = b_ptr + (
        offsets_k[:, None] * stride_bk + offsets_bn[None, :] * stride_bn
    )  # BLOCK_SIZE_Kx1 + 1xBLOCK_SIZE_N broadcast to BLOCK_SIZE_KxBLOCK_SIZE_N

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        mask_a = offsets_k[None, :] < K - k * BLOCK_SIZE_K
        mask_b = offsets_k[:, None] < K - k * BLOCK_SIZE_K
        a = tl.load(a_ptrs, mask=mask_a, other=0.0)
        b = tl.load(b_ptrs, mask=mask_b, other=0.0)
        accumulator += tl.dot(a, b)
        a_ptrs += stride_ak * BLOCK_SIZE_K
        b_ptrs += stride_bk * BLOCK_SIZE_K
    c = accumulator.to(tl.float16)

    offsets_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offsets_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + (
        offsets_cm[:, None] * stride_cm + offsets_cn[None, :] * stride_cn
    )  # BLOCK_SIZE_Mx1 + 1xBLOCK_SIZE_N broadcast to BLOCK_SIZE_MxBLOCK_SIZE_N
    mask_c = (offsets_cm[:, None] < M) & (offsets_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=mask_c)


def matmul(a: torch.Tensor, b: torch.Tensor):
    assert a.shape[1] == b.shape[0], "In compatible dimensions."
    assert a.is_contiguous() and b.is_contiguous(), "Matrix A and B must be contiguous."
    M, K = a.shape
    K, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
    )
    matmul_kernel[grid](
        a,
        b,
        c,
        M,
        N,
        K,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        c.stride(0),
        c.stride(1),
    )
    return c


def op_test():
    torch.manual_seed(0)
    a = torch.randn((512, 256), device=device, dtype=dtype)
    b = torch.randn((256, 512), device=device, dtype=dtype)
    c_torch = torch.matmul(a, b)
    c_triton = matmul(a, b)
    print(
        f"The maximum difference between torch and triton is "
        f"{torch.max(torch.abs(c_torch - c_triton))}"
    )


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["M", "N", "K"],  # Argument names to use as an x-axis for the plot
        x_vals=[
            128 * i for i in range(2, 33)
        ],  # Different possible values for `x_name`
        line_arg="provider",  # Argument name whose value corresponds to a different line in the plot
        # Possible values for `line_arg`
        line_vals=["torch", "triton", "paddle"],
        # Label name for the lines
        line_names=["Torch", "Triton", "Paddle"],
        # Line styles
        styles=[("green", "-"), ("blue", "-"), ("red", "-")],
        ylabel="ms",  # Label name for the y-axis
        plot_name="matmul-performance",  # Name for the plot, used also as a file name for saving the plot.
        args={},
    )
)
def benchmark(M, N, K, provider):
    a = torch.randn((M, K), device=device, dtype=dtype)
    b = torch.randn((K, N), device=device, dtype=dtype)
    a_p = paddle.to_tensor(
        np.random.randn(M, K), dtype="float16", place=paddle.CUDAPlace(0)
    )
    b_p = paddle.to_tensor(
        np.random.randn(M, K), dtype="float16", place=paddle.CUDAPlace(0)
    )
    quantiles = [0.5, 0.2, 0.8]
    if provider == "torch":
        ms, min_ms, max_ms = triton.testing.do_bench(
            fn=lambda: torch.matmul(a, b), quantiles=quantiles
        )
    if provider == "triton":
        ms, min_ms, max_ms = triton.testing.do_bench(
            fn=lambda: matmul(a, b), quantiles=quantiles
        )
    if provider == "paddle":
        ms, min_ms, max_ms = triton.testing.do_bench(
            fn=lambda: paddle.matmul(a_p, b_p), quantiles=quantiles
        )
    return ms, min_ms, max_ms


if __name__ == "__main__":
    op_test()
    benchmark.run(save_path="./perf_t4_cuda11.7_cudnn8.4", print_data=True)
