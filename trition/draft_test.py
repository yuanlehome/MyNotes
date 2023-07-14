import torch
import paddle

device = "cuda:0"

# x = torch.rand(size=(4, 5, 6), device=device)
# print(x.stride(), x.stride(0), x.stride(1), x.stride(2))
# print(x.max(dim=1)[0], x.max(dim=1)[0][:, None] < -0.5 + 2)
# # print(-float("inf"))

# print(x.t().shape)

BLOCK_SIZE_M = 128
BLOCK_SIZE_N = 128
BLOCK_SIZE_K = 1

x = torch.rand(size=(BLOCK_SIZE_M, BLOCK_SIZE_K))
y = torch.rand(size=(BLOCK_SIZE_K, BLOCK_SIZE_N))
z = torch.mul(x, y)
print(
    x.shape,
    y.shape,
    z.shape,
)
for k in range(0, 32, 8):
    print(k)
