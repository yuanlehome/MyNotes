import torch

import triton
import triton.language as tl


device = "cuda:0"


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


torch.manual_seed(0)
a = torch.randn((512, 256), device=device, dtype=torch.float16)
b = torch.randn((256, 512), device=device, dtype=torch.float16)
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
        line_vals=["cublas", "triton"],
        # Label name for the lines
        line_names=["cuBLAS", "Triton"],
        # Line styles
        styles=[("green", "-"), ("blue", "-")],
        ylabel="TFLOPS",  # Label name for the y-axis
        plot_name="matmul-performance",  # Name for the plot, used also as a file name for saving the plot.
        args={},
    )
)
def benchmark(M, N, K, provider):
    a = torch.randn((M, K), device=device, dtype=torch.float16)
    b = torch.randn((K, N), device=device, dtype=torch.float16)
    quantiles = [0.5, 0.2, 0.8]
    if provider == "cublas":
        ms, min_ms, max_ms = triton.testing.do_bench(
            fn=lambda: torch.matmul(a, b), quantiles=quantiles
        )
    if provider == "triton":
        ms, min_ms, max_ms = triton.testing.do_bench(
            fn=lambda: matmul(a, b), quantiles=quantiles
        )
    gbps = lambda ms: 2 * M * N * K * 1e-12 / (ms * 1e-3)
    return gbps(ms), gbps(max_ms), gbps(min_ms)


benchmark.run(show_plots=True, print_data=True)
# matmul-performance:
#          M     cuBLAS     Triton
# 0    256.0   3.216491   2.563755
# 1    384.0   8.869534   6.291456
# 2    512.0  15.155569   8.962188
# 3    640.0  21.250324  11.578799
# 4    768.0  30.442530  17.055152
# 5    896.0  27.296719  13.016125
# 6   1024.0  38.065152  17.019747
# 7   1152.0  32.042754  13.985874
# 8   1280.0  34.711863  16.451864
# 9   1408.0  38.016309  15.747334
# 10  1536.0  43.753969  17.698177
# 11  1664.0  36.474373  16.612738
# 12  1792.0  36.777092  19.071083
# 13  1920.0  38.858749  17.191023
# 14  2048.0  36.954221  16.876364
# 15  2176.0  36.273123  16.252504
# 16  2304.0  31.697951  15.688289
# 17  2432.0  34.076486  15.612373
# 18  2560.0  34.585923  15.774346
# 19  2688.0  33.153739  16.026429
# 20  2816.0  26.818883  15.159531
# 21  2944.0  29.333108  14.797197
# 22  3072.0  29.277716  15.142525
# 23  3200.0  20.815546  15.163519
# 24  3328.0  22.942690  14.939828
# 25  3456.0  20.781684  14.847776
# 26  3584.0  21.061785  12.549690
# 27  3712.0  20.992902  14.422192
# 28  3840.0  23.016174  14.097975
# 29  3968.0  22.275461  12.880427
# 30  4096.0  25.343227  13.639948
