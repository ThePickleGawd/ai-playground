import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button
from torchvision import transforms
import io
from model import Model

# Load model
model = Model()
state_dict = torch.load("checkpoints/mnist.pth", map_location="cpu")
model.load_state_dict(state_dict)
model.eval()

# Set up drawing canvas
canvas_size = 280  # 10x the MNIST size
img = np.ones((canvas_size, canvas_size), dtype=np.float32)

fig, ax = plt.subplots()
fig.canvas.manager.set_window_title('Draw a Digit')
plt.subplots_adjust(bottom=0.2)
im = ax.imshow(img, cmap='gray')
drawing = False

def on_press(event):
    global drawing
    drawing = True

def on_release(event):
    global drawing
    drawing = False

def on_motion(event):
    if drawing and event.xdata and event.ydata:
        x, y = int(event.xdata), int(event.ydata)
        size = 8
        img[y-size:y+size, x-size:x+size] = 0.0
        im.set_data(img)
        fig.canvas.draw()

def on_clear(event):
    global img
    img[:] = 1.0
    im.set_data(img)
    fig.canvas.draw()

def on_predict(event):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((28, 28)),
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    processed = transform(1.0 - img)  # Invert colors
    with torch.no_grad():
        output = model(processed.unsqueeze(0))
        pred = output.argmax(1).item()
    print(f"Prediction: {pred}")

# Add buttons
ax_clear = plt.axes([0.1, 0.05, 0.2, 0.075])
ax_predict = plt.axes([0.7, 0.05, 0.2, 0.075])
b_clear = Button(ax_clear, 'Clear')
b_clear.on_clicked(on_clear)
b_predict = Button(ax_predict, 'Predict')
b_predict.on_clicked(on_predict)

# Connect events
fig.canvas.mpl_connect('button_press_event', on_press)
fig.canvas.mpl_connect('button_release_event', on_release)
fig.canvas.mpl_connect('motion_notify_event', on_motion)

plt.show()
