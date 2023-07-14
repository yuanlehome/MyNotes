import triton
import triton.language as tl
import torch

device = "cuda:0"
dtype = torch.float16  # for benchmark


@triton.jit
def tl_dot_kernel(
    a_ptr, b_ptr, c_ptr, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr
):
    offsets_am = tl.arange(0, M)
    offsets_bn = tl.arange(0, N)
    offsets_k = tl.arange(0, K)
    a_ptrs = a_ptr + (offsets_am[:, None] * K + offsets_k[None, :] * 1)
    b_ptrs = b_ptr + (offsets_k[:, None] * N + offsets_bn[None, :] * 1)
    a = tl.load(a_ptrs)
    b = tl.load(b_ptrs)
    c = tl.dot(a, b)
    c_ptrs = c_ptr + (offsets_am[:, None] * N + offsets_bn[None, :] * 1)
    tl.store(c_ptrs, c)


def tl_dot(a, b):
    M, K = a.shape
    K, N = b.shape
    assert triton.next_power_of_2(M) == M, "Matrix size mismatch."
    assert triton.next_power_of_2(K) == K, "Matrix size mismatch."
    assert triton.next_power_of_2(N) == N, "Matrix size mismatch."

    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    grid = (1,)
    tl_dot_kernel[grid](a, b, c, M, N, K)
    return c


def test_tl_dot():
    torch.manual_seed(0)
    M, N, K = 128, 128, 128
    a = torch.randn((M, K), device=device, dtype=dtype)
    b = torch.randn((K, N), device=device, dtype=dtype)
    c_triton = tl_dot(a, b)
    c_torch = torch.matmul(a, b)
    print(
        f"The maximum difference between torch and triton is "
        f"{torch.max(torch.abs(c_torch - c_triton))}"
    )


if __name__ == "__main__":
    test_tl_dot()
    # Conclusion:
    # 1. triton maximum tensor numel (128x128=131072).
    # 2. tl.dot returns the matrix multiply of two blocks.
    # 3. tl.dot can perform a matrix multiply with block maximum size of 128x128.
