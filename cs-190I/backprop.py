import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        
        
        self.fc1 = nn.Linear(2, 2, bias=False)
        self.fc2 = nn.Linear(2, 2, bias=False)
        self.fc3 = nn.Linear(2, 2, bias=False)

        self.fc1.weight.data = torch.tensor([[1,-2],[-1,1]], dtype=torch.float32)
        self.fc2.weight.data = torch.tensor([[2,-1],[-2,-1]], dtype=torch.float32)
        self.fc3.weight.data = torch.tensor([[3,-1],[-1,4]], dtype=torch.float32)

    def forward(self, x1, x2):
        x = torch.tensor([x1,x2], dtype=torch.float32)

        x = self.fc1(x)
        x = F.relu(x)

        x = self.fc2(x)
        x = F.relu(x)

        out = self.fc3(x)
        return out
    
model = Model()
y1, y2 = model(1,-1)
yhat1, yhat2 = 21,-5

loss = 0.5 * torch.sum((yhat1 - y1)**2 + (yhat2 - y2)**2)

loss.backward()

for name, param in model.named_parameters():
    if param.grad is not None:
        print(f"{name}:\n{param.grad}")