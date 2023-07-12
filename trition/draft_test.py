import torch
import paddle

device = "cuda:0"

x = torch.rand(size=(4, 5), device=device)
# print(x.stride(), x.stride(0))
print(x.max(dim=1)[0], x.max(dim=1)[0][:, None] < -0.5 + 2)
# print(-float("inf"))

print(x.t().shape)