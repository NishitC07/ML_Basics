import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import matplotlib.pyplot as plt


# ----------- Config ---------
NUM_CLASSES = 2
BATCH_SIZE = 8
EPOCHS = 5
LR = 0.001
DEVICE = torch.device("cpu")
# ----------------------------

# 1. Load Pretrained ResNet18
model = models.resnet18(pretrained=True)

# 2. Freeze early layers (Feature Extraction)
for param in model.parameters():
    param.requires_grad = False

# 3. Replace final classification layer
model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)  # new output layer
model = model.to(DEVICE)

# 4. Simulate a small dataset (can use your own)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

dataset = datasets.ImageFolder(r"C:\Users\chaud\.cache\kagglehub\datasets\navoneel\brain-mri-images-for-brain-tumor-detection\versions\1\brain_tumor_dataset", transform=transform)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)



train_size = int(0.8 * len(dataset))  # 80% for training
test_size = len(dataset) - train_size

train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)


images, labels = next(iter(train_loader))
class_names = train_loader.dataset.dataset.classes

print(class_names)
# print(images[0])

# plt.imshow(images[0].permute(1, 2, 0).numpy())
# plt.axis('off')
# plt.title(class_names[labels[0]])
# plt.show()


criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.fc.parameters(), lr=LR)

# 6. Training Loop
for epoch in range(EPOCHS):
    total_loss = 0
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

        outputs = model(imgs)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {total_loss/len(train_loader):.4f}")


def evaluate(model, dataloader):
    model.eval()  # set to evaluation mode
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)  # shape: [batch, num_classes]
            _, predicted = torch.max(outputs, 1)  # get class with highest score
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    acc = 100 * correct / total
    print(f"Test Accuracy: {acc:.2f}%")
    model.train()  # switch back to training mode

evaluate(model, test_loader)
