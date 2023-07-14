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
                "BLOCK_SIZE_K": 32,
                "GROUP_SIZE_M": 8,
            },
            num_stages=2,
            num_warps=8,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_M": 256,
                "BLOCK_SIZE_N": 128,
                "BLOCK_SIZE_K": 32,
                "GROUP_SIZE_M": 8,
            },
            num_stages=2,
            num_warps=8,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_M": 256,
                "BLOCK_SIZE_N": 64,
                "BLOCK_SIZE_K": 32,
                "GROUP_SIZE_M": 8,
            },
            num_stages=2,
            num_warps=4,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_M": 64,
                "BLOCK_SIZE_N": 256,
                "BLOCK_SIZE_K": 32,
                "GROUP_SIZE_M": 8,
            },
            num_stages=2,
            num_warps=4,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_M": 128,
                "BLOCK_SIZE_N": 128,
                "BLOCK_SIZE_K": 32,
                "GROUP_SIZE_M": 8,
            },
            num_stages=2,
            num_warps=4,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_M": 128,
                "BLOCK_SIZE_N": 64,
                "BLOCK_SIZE_K": 32,
                "GROUP_SIZE_M": 8,
            },
            num_stages=3,
            num_warps=4,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_M": 64,
                "BLOCK_SIZE_N": 128,
                "BLOCK_SIZE_K": 32,
                "GROUP_SIZE_M": 8,
            },
            num_stages=3,
            num_warps=4,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_M": 128,
                "BLOCK_SIZE_N": 32,
                "BLOCK_SIZE_K": 32,
                "GROUP_SIZE_M": 8,
            },
            num_stages=3,
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
    # Pointers to matrices
    a_ptr,
    b_ptr,
    c_ptr,
    # Matrix dimensions
    M,
    N,
    K,
    # The stride variables represent how much to increase the ptr by when moving by 1
    # element in a particular dimension. E.g. stride_am is how much to increase a_ptr
    # by to get the element one row down (A has M rows)
    stride_a_batch,
    stride_am,
    stride_ak,
    stride_b_batch,
    stride_bk,
    stride_bn,
    stride_c_batch,
    stride_cm,
    stride_cn,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    """Kernel for computing the matmul C = A x B.
    A has shape (M, K), B has shape (K, N) and C has shape (M, N)
    """
    # -----------------------------------------------------------
    # Map program ids `pid` to the block of C it should compute.
    # This is done in a grouped ordering to promote L2 data reuse
    # See above `L2 Cache Optimizations` section for details

    # Supergrouping of blocks
    # To see later
    batch_id = tl.program_id(axis=1)
    # program ID
    pid = tl.program_id(axis=0)

    # number of program ids along the M axis
    grid_m = tl.cdiv(M, BLOCK_SIZE_M)
    # number of programs ids along the N axis
    grid_n = tl.cdiv(N, BLOCK_SIZE_N)

    # number of programs in group
    num_pid_in_group = GROUP_SIZE_M * grid_n
    # id of the group this program is in
    group_id = pid // num_pid_in_group
    # row-id of the first program in the group
    first_pid_m = group_id * GROUP_SIZE_M
    # if `grid_m` isn't divisible by `GROUP_SIZE_M`, the last group is smaller
    group_size_m = min(grid_m - first_pid_m, GROUP_SIZE_M)
    # *within groups*, programs are ordered in a column-major order
    # row-id of the program in the *launch grid*
    pid_m = first_pid_m + (pid % group_size_m)
    # col-id of the program in the *launch grid*
    pid_n = (pid % num_pid_in_group) // group_size_m

    # ----------------------------------------------------------
    # Create pointers for the first blocks of A and B.
    # We will advance this pointer as we move in the K direction
    # and accumulate
    # a_ptrs is a block of [BLOCK_SIZE_M, BLOCK_SIZE_K] pointers
    # b_ptrs is a block of [BLOCK_SIZE_K, BLOCK_SIZE_N] pointers
    # see above `Pointer Arithmetics` section for details

    # pid_m * BLOCK_SIZE_M is the row index of the first element of the block of size BLOCK_SIZE_M
    # We add tl.arange(0, BLOCK_SIZE_M) to get a vector of row indexes
    offsets_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offsets_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offsets_k = tl.arange(0, BLOCK_SIZE_K)

    # a_offs[:, None] is a column vector of BLOCK_SIZE_M rows indexes
    # We multiply by stride_am, to we get a column vector of memory offsets to each start of a row
    # k_range_offs[None, :] is a row vector of size BLOCK_SIZE_K columns indexes
    # We multiply stride_ak to get a row vector of memory offsets to each start of a column
    # When we add both. We get a matrix of memory offsets.
    # For A in RowMajor stride_ak will be 1, so k_range_offs[None, :] * stride_ak will be
    # just 0,1,2,3,4,5....BLOCK_SIZE_K
    a_ptrs = (
        a_ptr
        + stride_a_batch * batch_id
        + (offsets_am[:, None] * stride_am + offsets_k[None, :] * stride_ak)
    )  # BLOCK_SIZE_Mx1 + 1xBLOCK_SIZE_K broadcast to BLOCK_SIZE_MxBLOCK_SIZE_K
    b_ptrs = (
        b_ptr
        + stride_b_batch * batch_id
        + (offsets_k[:, None] * stride_bk + offsets_bn[None, :] * stride_bn)
    )  # BLOCK_SIZE_Kx1 + 1xBLOCK_SIZE_N broadcast to BLOCK_SIZE_KxBLOCK_SIZE_N

    # -----------------------------------------------------------
    # Iterate to compute a block of the C matrix
    # We accumulate into a `[BLOCK_SIZE_M, BLOCK_SIZE_N]` block
    # of fp32 values for higher accuracy.
    # `accumulator` will be converted back to fp16 after the loop
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        mask_a = offsets_k[None, :] < K - k * BLOCK_SIZE_K
        mask_b = offsets_k[:, None] < K - k * BLOCK_SIZE_K
        a = tl.load(a_ptrs, mask=mask_a, other=0.0)
        b = tl.load(b_ptrs, mask=mask_b, other=0.0)
        # We accumulate along the K dimension
        accumulator += tl.dot(a, b)
        # Advance the ptrs to the next K block
        a_ptrs += stride_ak * BLOCK_SIZE_K
        b_ptrs += stride_bk * BLOCK_SIZE_K

    c = accumulator.to(tl.float16)

    # -----------------------------------------------------------
    # Write back the block of the output matrix C
    offsets_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offsets_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = (
        c_ptr
        + stride_c_batch * batch_id
        + (offsets_cm[:, None] * stride_cm + offsets_cn[None, :] * stride_cn)
    )  # BLOCK_SIZE_Mx1 + 1xBLOCK_SIZE_N broadcast to BLOCK_SIZE_MxBLOCK_SIZE_N
    mask_c = (offsets_cm[:, None] < M) & (offsets_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=mask_c)


def batched_matmul(a, b):
    # checks constraints
    assert a.shape[0] == b.shape[0], "Batch dimension of A must be the same as B."
    assert a.shape[2] == b.shape[1], "Incompatible dimensions."
    assert a.is_contiguous(), "Matrix A must be contiguous."
    assert b.is_contiguous(), "Matrix B must be contiguous."
    batch_size, M, K = a.shape
    _, K, N = b.shape
    # allocates output
    c = torch.empty((batch_size, M, N), device=a.device, dtype=a.dtype)
    # 1D launch kernel where each block gets its own program.

    grid = lambda META: (  # noqa: E731
        triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
        batch_size,
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
        a.stride(2),
        b.stride(0),
        b.stride(1),
        b.stride(2),
        c.stride(0),
        c.stride(1),
        c.stride(2),
    )
    return c


def op_test():
    torch.manual_seed(0)
    a = torch.randn((4, 512, 256), device=device, dtype=dtype)
    b = torch.randn((4, 256, 512), device=device, dtype=dtype)
    c_torch = torch.matmul(a, b)
    c_triton = batched_matmul(a, b)
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
        plot_name="matmul-batched-performance",  # Name for the plot, used also as a file name for saving the plot.
        args={
            "Batch": 8,
        },
    )
)
def benchmark(Batch, M, N, K, provider):
    a = torch.randn((Batch, M, K), device=device, dtype=dtype)
    b = torch.randn((Batch, K, N), device=device, dtype=dtype)
    a_p = paddle.to_tensor(
        np.random.randn(Batch, M, K), dtype="float16", place=paddle.CUDAPlace(0)
    )
    b_p = paddle.to_tensor(
        np.random.randn(Batch, M, K), dtype="float16", place=paddle.CUDAPlace(0)
    )
    quantiles = [0.5, 0.2, 0.8]
    if provider == "torch":
        ms, min_ms, max_ms = triton.testing.do_bench(
            fn=lambda: torch.matmul(a, b), quantiles=quantiles
        )
    if provider == "triton":
        ms, min_ms, max_ms = triton.testing.do_bench(
            fn=lambda: batched_matmul(a, b), quantiles=quantiles
        )
    if provider == "paddle":
        ms, min_ms, max_ms = triton.testing.do_bench(
            fn=lambda: paddle.matmul(a_p, b_p), quantiles=quantiles
        )
    return ms, min_ms, max_ms


if __name__ == "__main__":
    op_test()
    # benchmark.run(save_path="./perf_t4_cuda11.7_cudnn8.4", print_data=True)
