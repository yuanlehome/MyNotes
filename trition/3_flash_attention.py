import pytest
import torch

import triton
import triton.language as tl


device = "cuda:0"
dtype = torch.float16  # for benchmark


@triton.jit
def fwd_kernel(
    Q,
    K,
    V,
    sm_scale,
    L,
    M,
    Out,
    stride_qz,
    stride_qh,
    stride_qm,
    stride_qk,
    stride_kz,
    stride_kh,
    stride_kn,
    stride_kk,
    stride_vz,
    stride_vh,
    stride_vk,
    stride_vn,
    stride_oz,
    stride_oh,
    stride_om,
    stride_on,
    Z,
    H,
    N_CTX,
    BLOCK_M: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
    MODE: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    qvk_offset = off_hz * stride_qh
    Q_block_ptr = tl.make_block_ptr(
        base=Q + qvk_offset,
        shape=(N_CTX, BLOCK_DMODEL),
        strides=(stride_qm, stride_qk),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL),
        order=(1, 0),
    )
    K_block_ptr = tl.make_block_ptr(
        base=K + qvk_offset,
        shape=(BLOCK_DMODEL, N_CTX),
        strides=(stride_kk, stride_kn),
        offsets=(0, 0),
        block_shape=(BLOCK_DMODEL, BLOCK_N),
        order=(0, 1),
    )
    V_block_ptr = tl.make_block_ptr(
        base=V + qvk_offset,
        shape=(N_CTX, BLOCK_DMODEL),
        strides=(stride_vk, stride_vn),
        offsets=(0, 0),
        block_shape=(BLOCK_N, BLOCK_DMODEL),
        order=(1, 0),
    )
    O_block_ptr = tl.make_block_ptr(
        base=Out + qvk_offset,
        shape=(N_CTX, BLOCK_DMODEL),
        strides=(stride_om, stride_on),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL),
        order=(1, 0),
    )
    # initialize offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    # initialize pointer to m and l
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    # causal check on every loop iteration can be expensive
    # and peeling the last iteration of the loop does not work well with ptxas
    # so we have a mode to do the causal check in a separate kernel entirely
    if MODE == 0:  # entire non-causal attention
        lo, hi = 0, N_CTX
    if MODE == 1:  # entire causal attention
        lo, hi = 0, (start_m + 1) * BLOCK_M
    if MODE == 2:  # off band-diagonal
        lo, hi = 0, start_m * BLOCK_M
    if MODE == 3:  # on band-diagonal
        l_ptrs = L + off_hz * N_CTX + offs_m
        m_ptrs = M + off_hz * N_CTX + offs_m
        m_i = tl.load(m_ptrs)
        l_i = tl.load(l_ptrs)
        acc += tl.load(O_block_ptr).to(tl.float32)
        lo, hi = start_m * BLOCK_M, (start_m + 1) * BLOCK_M
    # credits to: Adam P. Goucher (https://github.com/apgoucher):
    # scale sm_scale by 1/log_2(e) and use
    # 2^x instead of exp in the loop because CSE and LICM
    # don't work as expected with `exp` in the loop
    qk_scale = sm_scale * 1.44269504
    # load q: it will stay in SRAM throughout
    q = tl.load(Q_block_ptr)
    q = (q * qk_scale).to(tl.float16)
    # loop over k, v and update accumulator
    for start_n in range(lo, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        # -- compute qk ----
        k = tl.load(tl.advance(K_block_ptr, (0, start_n)))
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk += tl.dot(q, k)
        if MODE == 1 or MODE == 3:
            qk = tl.where(
                offs_m[:, None] >= (start_n + offs_n[None, :]), qk, float("-inf")
            )
        # -- compute m_ij, p, l_ij
        m_ij = tl.max(qk, 1)
        p = tl.math.exp2(qk - m_ij[:, None])
        l_ij = tl.sum(p, 1)
        # -- update m_i and l_i
        m_i_new = tl.maximum(m_i, m_ij)
        alpha = tl.math.exp2(m_i - m_i_new)
        beta = tl.math.exp2(m_ij - m_i_new)
        l_i *= alpha
        l_i_new = l_i + beta * l_ij
        # scale p
        p_scale = beta / l_i_new
        p = p * p_scale[:, None]
        # scale acc
        acc_scale = l_i / l_i_new
        acc = acc * acc_scale[:, None]
        # update acc
        v = tl.load(tl.advance(V_block_ptr, (start_n, 0)))
        p = p.to(tl.float16)
        acc += tl.dot(p, v)
        # update m_i and l_i
        l_i = l_i_new
        m_i = m_i_new
    # write back l and m
    l_ptrs = L + off_hz * N_CTX + offs_m
    m_ptrs = M + off_hz * N_CTX + offs_m
    tl.store(l_ptrs, l_i)
    tl.store(m_ptrs, m_i)
    # write back O
    tl.store(O_block_ptr, acc.to(tl.float16))


def attention(q, k, v, causal, sm_scale):
    BLOCK = 128
    # shape constraints
    Lq, Lk, Lv = q.shape[-1], k.shape[-1], v.shape[-1]
    assert Lq == Lk and Lk == Lv
    assert Lk in {16, 32, 64, 128}
    o = torch.empty_like(q)
    grid = (triton.cdiv(q.shape[2], 128), q.shape[0] * q.shape[1], 1)
    L = torch.empty(
        (q.shape[0] * q.shape[1], q.shape[2]), device=q.device, dtype=torch.float32
    )
    m = torch.empty(
        (q.shape[0] * q.shape[1], q.shape[2]), device=q.device, dtype=torch.float32
    )

    num_warps = 4 if Lk <= 64 else 8
    if causal:
        modes = [1] if q.shape[2] <= 2048 else [2, 3]
    else:
        modes = [0]
    for mode in modes:
        fwd_kernel[grid](
            q,
            k,
            v,
            sm_scale,
            L,
            m,
            o,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            q.stride(3),
            k.stride(0),
            k.stride(1),
            k.stride(2),
            k.stride(3),
            v.stride(0),
            v.stride(1),
            v.stride(2),
            v.stride(3),
            o.stride(0),
            o.stride(1),
            o.stride(2),
            o.stride(3),
            q.shape[0],
            q.shape[1],
            q.shape[2],
            BLOCK_M=128,
            BLOCK_N=BLOCK,
            BLOCK_DMODEL=Lk,
            MODE=mode,
            num_warps=num_warps,
            num_stages=2,
        )

    return o


def op_test():
    Z, H, N_CTX, D_HEAD = [6, 9, 1024, 64]
    causal = True
    torch.manual_seed(20)
    q = torch.empty((Z, H, N_CTX, D_HEAD), dtype=dtype, device="cuda").normal_(
        mean=0.0, std=0.5
    )
    k = torch.empty((Z, H, N_CTX, D_HEAD), dtype=dtype, device=device).normal_(
        mean=0.0, std=0.5
    )
    v = torch.empty((Z, H, N_CTX, D_HEAD), dtype=dtype, device=device).normal_(
        mean=0.0, std=0.5
    )
    sm_scale = 0.5
    # reference implementation
    M = torch.tril(torch.ones((N_CTX, N_CTX), device=device))
    p = torch.matmul(q, k.transpose(2, 3)) * sm_scale
    if causal:
        p[:, :, M == 0] = float("-inf")
    p = torch.softmax(p.float(), dim=-1).half()
    # p = torch.exp(p)
    ref_out = torch.matmul(p, v)

    # triton implementation
    tri_out = attention(q, k, v, causal, sm_scale).half()
    # compare
    assert torch.allclose(ref_out, tri_out, atol=1e-2, rtol=0)


try:
    from flash_attn.flash_attn_interface import flash_attn_func

    HAS_FLASH = True
except BaseException:
    HAS_FLASH = False

BATCH, N_HEADS, N_CTX, D_HEAD = 4, 48, 4096, 64
# vary seq length for fixed head and batch=4
@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["N_CTX"],
        x_vals=[2**i for i in range(10, 15)],
        line_arg="provider",
        line_vals=["triton"] + (["flash"] if HAS_FLASH else []),
        line_names=["Triton"] + (["Flash"] if HAS_FLASH else []),
        styles=[("red", "-"), ("blue", "-")],
        ylabel="ms",
        plot_name=f"fused-attention-batch{BATCH}-head{N_HEADS}-d{D_HEAD}-{mode}-causal:{causal}",
        args={
            "H": N_HEADS,
            "BATCH": BATCH,
            "D_HEAD": D_HEAD,
            "dtype": torch.float16,
            "mode": mode,
            "causal": causal,
        },
    )
    for mode in ["fwd"]
    for causal in [False, True]
)
def benchmark(
    BATCH, H, N_CTX, D_HEAD, causal, mode, provider, dtype=torch.float16, device=device
):
    warmup = 25
    rep = 100
    if provider == "triton":
        q = torch.randn((BATCH, H, N_CTX, D_HEAD), dtype=dtype, device=device)
        k = torch.randn((BATCH, H, N_CTX, D_HEAD), dtype=dtype, device=device)
        v = torch.randn((BATCH, H, N_CTX, D_HEAD), dtype=dtype, device=device)
        sm_scale = 1.3
        ms = triton.testing.do_bench(
            fn=lambda: attention(q, k, v, causal, sm_scale), warmup=warmup, rep=rep
        )
    if provider == "flash":
        lengths = torch.full((BATCH,), fill_value=N_CTX, device=device)
        cu_seqlens = torch.zeros((BATCH + 1,), device=device, dtype=torch.int32)
        cu_seqlens[1:] = lengths.cumsum(0)
        qkv = torch.randn((BATCH * N_CTX, 3, H, D_HEAD), dtype=dtype, device=device)
        ms = triton.testing.do_bench(
            fn=lambda: flash_attn_func(qkv, cu_seqlens, 0.0, N_CTX, causal=True),
            warmup=warmup,
            rep=rep,
        )
    return ms


if __name__ == "__main__":
    op_test()
    # only works on post-Ampere GPUs right now
    benchmark.run(save_path="./perf_a10", print_data=True)
