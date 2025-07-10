'''
Homework:
Methodology for solving image classification problems.
Train a simple convolutional neural network (CNN) to classify CIFAR images.
'''

# %%
# importing
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torchvision
import numpy as np
from matplotlib import pyplot as plt
import random

# %%
# load CIFAR-10 data
transform = transforms.Compose([
    transforms.ToTensor(),  # Converts to [0,1]
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize to [-1,1]
])

train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# %%
# inspect images
classes = train_dataset.classes

def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

dataiter = iter(train_loader)
images, labels = next(dataiter)
imshow(torchvision.utils.make_grid(images[:4]))
print(' '.join(f'{classes[labels[j]]}' for j in range(4)))

# %%
# create a CNN model
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 64 * 8 * 8)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleCNN().to(device)

# %%
# train the model
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
epochs = 5


for epoch in range(epochs):
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader):.4f}")

# %%
# evaluate the model
correct = 0
total = 0
model.eval()
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
print(f"Test Accuracy: {100 * correct / total:.2f}%")

# Show a batch of test images with predictions
def show_predictions(model, loader, classes, device, num_images=12):
    model.eval()
    images_shown = 0
    plt.figure(figsize=(12, 6))
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            for i in range(images.size(0)):
                if images_shown >= num_images:
                    break
                img = images[i].cpu() / 2 + 0.5  # unnormalize
                npimg = img.numpy()
                plt.subplot(3, 4, images_shown + 1)
                plt.imshow(np.transpose(npimg, (1, 2, 0)))
                color = 'green' if preds[i] == labels[i] else 'red'
                plt.title(f"P: {classes[preds[i]]}\nT: {classes[labels[i]]}", color=color, fontsize=10)
                plt.axis('off')
                images_shown += 1
            if images_shown >= num_images:
                break
    plt.tight_layout()
    plt.suptitle("CIFAR-10 Predictions (Green=Correct, Red=Incorrect)", fontsize=16, y=1.05)
    plt.show()

show_predictions(model, test_loader, classes, device)