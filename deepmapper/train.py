import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision import transforms, datasets
from model import Model

device = torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")

print(f"Using device {device}")


transform = transforms.Compose([
    transforms.ToTensor()
])

train_dataset = datasets.MNIST('mnistfiles', train=True, transform=transform)
test_dataset = datasets.MNIST('mnistfiles', train=False, transform=transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64)


model = Model().to(device)
optim = torch.optim.AdamW(model.parameters())

epochs = 1
for epoch in range(epochs):
    for idx, (X,y) in enumerate(train_loader):
        X, y = X.to(device), y.to(device)
        logits = model(X)

        loss = F.cross_entropy(logits, y)
        loss.backward()
        optim.step()
        model.zero_grad()

        if idx % 10 == 0:
            print(f"Loss: {loss}")

    
    # Sample training accuracy
    correct = 0
    test_loss = 0
    with torch.no_grad():
        model.eval()
        for X, y in test_loader:
            X, y = X.to(device), y.to(device)
            logits = model(X)

            loss = F.cross_entropy(logits, y)
            test_loss += loss

            pred = logits.argmax(1)
            correct += torch.eq(pred, y).sum()

        print(f"Test Accuracy: {correct/len(test_loader.dataset)}, Test Loss: {test_loss/len(test_loader.dataset)}")

    print(f"Epoch: {epoch}")


print("Done. Saving to mnist.pth")
torch.save(model.state_dict(), "checkpoints/mnist.pth")