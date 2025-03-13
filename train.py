import torch
from model import GPT2
import os
import tiktoken

tokenizer = tiktoken.get_encoding("gpt2")

device = "cpu"
if torch.cuda.is_available():
    device = "gpu"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = "mps"

device = "cpu"
print(f"Using device {device}")

# Load data as an array of characters, then tokens
data_path = os.path.join("data", "shakespeare", "input.txt")
with open(data_path, "r") as f:
    text = f.read()

text = text[:1000]
tokens = tokenizer.encode(text)

# Data batch; y[i]=x[i+1] for labels
B, T = 4, 32
buf = torch.tensor(tokens[:B*T + 1])
x = buf[:-1].view(B,T)
y = buf[1:].view(B,T)

model = GPT2()
model.to(device)
logits = model(x)

epochs = 1

for _ in range(epochs):
    output, loss = model(x,y)
    print(loss)