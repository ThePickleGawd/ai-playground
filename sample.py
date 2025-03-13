import torch
import tiktoken
from model import GPT2
import sys

tokenizer = tiktoken.get_encoding("gpt2")

input = [
    "Dylan is cool",
    "My name is...", 
    "HELLO!"
]

input_tensor = torch.tensor(tokenizer.encode(input[0])).unsqueeze(dim=0)
model = GPT2()
model.eval()

with torch.no_grad():
    logits = model(input_tensor)
    logits = logits[-1, -1, :] # Get the last logit row, the numbers represent the "almost" probability of the next token
    print(logits.size())
    probs = torch.softmax(logits, dim=-1)
    token = torch.argmax(probs, dim=-1)
    print(token)
    res = tokenizer.decode([token.item()])
    print(res)