import torch
from flash_attn.flash_attention import FlashMHA

# Replace this with your correct GPU device
device = "cuda:0"

# Create attention layer. This is similar to torch.nn.MultiheadAttention,
# and it includes the input and output linear layers
flash_mha = FlashMHA(
    embed_dim=128,  # total channels (= num_heads * head_dim)
    num_heads=8,  # number of heads
    device=device,
    dtype=torch.float16,
)

# Run forward pass with dummy data
x = torch.randn(
    (64, 256, 128), device=device, dtype=torch.float16  # (batch, seqlen, embed_dim)
)

output = flash_mha(x)[0]

print(output)
