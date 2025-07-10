import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# 1. Load Data
transform = transforms.ToTensor()

train_data = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_data = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=64)

# 2. Build Model
class DigitClassifier(nn.Module):
    def __init__(self):
        super(DigitClassifier, self).__init__()
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28, 128),   # Input: 784 → Hidden: 128
            nn.ReLU(),
            nn.Linear(128, 10) # Output: 10 classes
        )

    def forward(self, x):
        return self.network(x)

model = DigitClassifier()

# 3. Loss + Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 4. Training Loop
for epoch in range(7):
    total_loss = 0
    for images, labels in train_loader:
        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    loss = total_loss/len(train_loader)
    print(f"Epoch {epoch+1}, Loss: {loss:.4f}")

# 5. Evaluate on Test Set
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Test Accuracy: {100 * correct / total:.2f}%")

# 6. Visualize some predictions
examples = iter(test_loader)
example_data, example_labels = next(examples)

with torch.no_grad():
    predictions = model(example_data)

fig = plt.figure(figsize=(10, 4))
for i in range(6):
    plt.subplot(2, 3, i+1)
    plt.imshow(example_data[i][0], cmap='gray')
    _, pred_label = torch.max(predictions[i], 0)
    plt.title(f"Predicted: {pred_label.item()}")
    plt.axis('off')
plt.tight_layout()
plt.show()
