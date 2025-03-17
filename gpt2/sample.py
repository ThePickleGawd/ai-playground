import torch
import tiktoken
from model import GPT2, GPTConfig
import sys
import os

## Device and model setup
device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = "mps"

print(f"Using device {device}")

enc = tiktoken.get_encoding("gpt2")
gptconfig = GPTConfig()
model = GPT2(gptconfig)
model.to(device)

checkpoint = "checkpoints/gpt2_shakespeare_model.pth"
if os.path.exists(checkpoint):
    model.load_state_dict(torch.load("checkpoints/gpt2_shakespeare_model.pth"))
else:
    print("No checkpoint available. This is an untrained model")
model.eval()

with torch.no_grad():
    out = model.generate(torch.tensor(enc.encode("DYLAN LU:"), dtype=torch.long, device=device).unsqueeze(dim=0), max_tokens=250)
    print(out.squeeze(0).tolist())
    print(enc.decode(out.squeeze(0).tolist()))