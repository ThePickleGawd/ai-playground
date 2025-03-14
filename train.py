import os
import torch
from torch.utils.data import DataLoader
from model import GPT2
from data.shakespeare.dataset import ShakespeareDataset
import tiktoken

enc = tiktoken.get_encoding("gpt2")

device = "cpu"
if torch.cuda.is_available():
    device = "gpu"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = "mps"

device = "cpu" # TODO: Remove hardcoded cpu
print(f"Using device {device}")


model = GPT2()
model.to(device)

train_dataset = ShakespeareDataset(train=True)
test_dataset = ShakespeareDataset(train=False)

train_dataloader = DataLoader(train_dataset, batch_size=12, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=12, shuffle=True)


epochs = 1

for _ in range(epochs):
    x,y = next(iter(train_dataloader))
    logits, loss = model(x, y)
    print(loss)