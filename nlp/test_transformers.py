import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer


model = GPT2LMHeadModel.from_pretrained("bert-base-uncased", torchscript=True).eval()

# tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("bert-base-uncased", resume_download=True)
in_text = "Lionel Messi is a"
in_tokens = torch.tensor(tokenizer.encode(in_text))

# inference
token_eos = torch.tensor([198])  # line break symbol
out_token = None
i = 0
with torch.no_grad():
    while out_token != token_eos:
        logits, _ = model(in_tokens)
        out_token = torch.argmax(logits[-1, :], dim=0, keepdim=True)
        in_tokens = torch.cat((in_tokens, out_token), 0)
        text = tokenizer.decode(in_tokens)
        print(f"step {i} input: {text}", flush=True)
        i += 1

out_text = tokenizer.decode(in_tokens)
print(f" Input: {in_text}")
print(f"Output: {out_text}")
