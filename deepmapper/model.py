# Dylan's playground with CNNs for the MNIST dataset

"""

(512, 512)

"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super().__init__()

        # 2x ConvBlocks, 3x FC
        # Input (B, 1, 28, 28)

        self.features = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1), # (B, 1, 13, 13)
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1), # (B, 16, 26, 26)
            nn.ReLU(),

            nn.Flatten(1),

            nn.Linear(in_features=18432, out_features=4096),
            nn.ReLU(),
            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU(),
            nn.Linear(in_features=4096, out_features=10)
        )
        
        
    def forward(self, X):
        B = X.shape[0]

        X = self.features(X)
        return X

        