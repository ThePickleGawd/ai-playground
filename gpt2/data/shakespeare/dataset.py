import os
import torch
import tiktoken
from torch.utils.data import Dataset
import numpy as np


class ShakespeareDataset(Dataset):
    def __init__(self, train=True, block_size=1024):
        enc = tiktoken.get_encoding("gpt2")

        self.block_size = block_size

        script_dir = os.path.dirname(__file__)
        if train:
            self.tokens = np.memmap(os.path.join(script_dir, 'train.bin'), dtype=np.uint16, mode='r')
        else:
            self.tokens = np.memmap(os.path.join(script_dir, 'val.bin'), dtype=np.uint16, mode='r') 

    def __len__(self):
        return (len(self.tokens) - 1) // self.block_size # Token's label is i+1; Only allow idx of batches

    def __getitem__(self, idx):
        start = idx * self.block_size
        x = torch.tensor(self.tokens[start: start + self.block_size], dtype=torch.long)
        y = torch.tensor(self.tokens[start + 1: start + self.block_size + 1], dtype=torch.long)
        return x, y
