import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(2, 2),
            nn.ReLU(),
            nn.Linear(2, 2),
            nn.ReLU()
        )

    def forward(self, x1, x2):
        return self.network(torch.tensor([x1,x2]))
    
model = Model()
print(model(1,2))