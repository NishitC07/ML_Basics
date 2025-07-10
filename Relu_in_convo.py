import torch
import torch.nn.functional as F

x = torch.randn(1, 3, 3)  # Output from a convolution layer
print(x)
x_activated = F.relu(x)      # Applies ReLU to every element
print(x_activated)