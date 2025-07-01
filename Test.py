import numpy as np

X = np.array([[1,3,4],
              [2,5,3],
              [4,3,2],
              [2,3,5]])

y = np.array([0,1,0,1])

w = np.array([0.32,-0.43,-0.23])
b = 0.34
lr = 0.01

print(y)
def sigmoid(z):
    return 1/(1+np.exp(-z))

def softmax(z):
    exp = np.exp(z - np.max(z, axis=1, keepdims=True))  # Stability trick
    return exp / np.sum(exp, axis=1, keepdims=True)

def BCE(y,y_hat):
    epsilon = 1e-9
    y_hat = np.clip(y_hat,epsilon, 1-epsilon)
    return -np.mean(y*np.log(y_hat) + (1-y)*np.log(1-y_hat))

def descent(x,y,w,b,lr):
    loss = 0
    dldw = 0
    dldb = 0
    N = x.shape[0]
    logits = x @ w + b
    prob = sigmoid(logits)

    error = y - prob

    dldw = (1/N) * (x.T @ error)
    dldb = (1/N) * np.sum(error)





















logits = X @ w + b
print("Raw Logits",logits)
prob = sigmoid(logits)
print("Probabilities",sigmoid(logits))
print("loss",BCE(y,prob))
