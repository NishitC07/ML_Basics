import numpy as np

def relu(x):
    return np.maximum(0, x)

def relu_deriv(x):
    return (x > 0).astype(float)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_deriv(x):
    sx = sigmoid(x)
    return sx * (1 - sx)

# XOR-type data
X = np.array([[0,0], [0,1], [1,0], [1,1]])
y = np.array([[0], [1], [1], [0]])

np.random.seed(42)
# Layer 1: 2 → 5
W1 = np.random.randn(2, 5)
b1 = np.zeros((1, 5))

# Layer 2: 5 → 3
W2 = np.random.randn(5, 3)
b2 = np.zeros((1, 3))

# Output: 3 → 1
W3 = np.random.randn(3, 1)
b3 = np.zeros((1, 1))

lr = 0.1

# Training loop
for epoch in range(15000):
    # FORWARD
    z1 = np.dot(X, W1) + b1
    a1 = relu(z1)

    z2 = np.dot(a1, W2) + b2
    a2 = sigmoid(z2)

    z3 = np.dot(a2, W3) + b3
    a3 = sigmoid(z3)  # final output

    # LOSS
    loss = np.mean((y - a3) ** 2)

    # BACKPROP
    dz3 = (a3 - y) * sigmoid_deriv(z3)
    dW3 = np.dot(a2.T, dz3)
    db3 = np.sum(dz3, axis=0, keepdims=True)

    dz2 = np.dot(dz3, W3.T) * sigmoid_deriv(z2)
    dW2 = np.dot(a1.T, dz2)
    db2 = np.sum(dz2, axis=0, keepdims=True)

    dz1 = np.dot(dz2, W2.T) * relu_deriv(z1)
    dW1 = np.dot(X.T, dz1)
    db1 = np.sum(dz1, axis=0, keepdims=True)

    # Update
    W3 -= lr * dW3
    b3 -= lr * db3
    W2 -= lr * dW2
    b2 -= lr * db2
    W1 -= lr * dW1
    b1 -= lr * db1

    if epoch % 1000 == 0:
        print(f"Epoch {epoch} | Loss: {loss:.4f}")

print("Final predictions:")
print(a3.round())
