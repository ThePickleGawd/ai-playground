import os
import torch
import tiktoken
from torch.utils.data import Dataset
import numpy as np


class ShakespeareDataset(Dataset):
    def __init__(self, train=True, block_size=8):
        enc = tiktoken.get_encoding("gpt2")

        self.block_size = block_size

        script_dir = os.path.dirname(__file__)
        if train:
            self.tokens = np.memmap(os.path.join(script_dir, 'train.bin'), dtype=np.uint16, mode='r')
        else:
            self.tokens = np.memmap(os.path.join(script_dir, 'val.bin'), dtype=np.uint16, mode='r') 

    def __len__(self):
        return len(self.tokens) - 1 # Every token's label is the next token, so the very last one won't have a label

    def __getitem__(self, idx):
        x = torch.tensor(self.tokens[idx: idx + self.block_size], dtype=torch.int64)
        y = torch.tensor(self.tokens[idx + 1: idx + self.block_size + 1], dtype=torch.int64)
        return x, y
