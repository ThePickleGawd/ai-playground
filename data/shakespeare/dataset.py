import os
import torch
import tiktoken
from torch.utils.data import Dataset
import numpy as np


class ShakespeareDataset(Dataset):
    def __init__(self, train=True):
        enc = tiktoken.get_encoding("gpt2")

        script_dir = os.path.dirname(__file__)
        if train:
            tokens = np.memmap(os.path.join(script_dir, 'train.bin'), dtype=np.uint16, mode='r')
        else:
            tokens = np.memmap(os.path.join(script_dir, 'val.bin'), dtype=np.uint16, mode='r') 

        # Set label of token to be the next token
        self.data = tokens[:-1]
        self.labels = tokens[1:]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]