import os
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from model import GPT2, GPTConfig
from data.shakespeare.dataset import ShakespeareDataset
import tiktoken


## Device and model setup
device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = "mps"

print(f"Using device {device}")

gptconf = GPTConfig()
model = GPT2(gptconf)
model.to(device)

## Data setup
train_dataset = ShakespeareDataset(train=True, block_size=gptconf.block_size)
test_dataset = ShakespeareDataset(train=False, block_size=gptconf.block_size)

train_dataloader = DataLoader(train_dataset, batch_size=12, shuffle=True, drop_last=True)
test_dataloader = DataLoader(test_dataset, batch_size=12, shuffle=True, drop_last=True)

enc = tiktoken.get_encoding("gpt2")

## Settings
epochs = 100
lr = 1e-3
optimizer = AdamW(model.parameters(), lr=lr)

for _ in range(epochs):
    model.train()

    iter_num = 0
    
    # Forward pass
    for x,y in train_dataloader:
        x, y = x.to(device), y.to(device)
        logits, loss = model(x, y)

        # Learn!
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        iter_num += 1
        if iter_num % 5 == 0:
            print(f"loss: {loss}")

    # Print sample
    out = model.generate(torch.tensor(enc.encode("DYLAN:\n"), dtype=torch.long, device=device).unsqueeze(dim=0))
    print(enc.decode(out.squeeze(0).tolist()))