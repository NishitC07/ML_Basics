import numpy as np

W = 4
B = np.random.rand() 
print(B)
X = np.random.randn(10,1)

Y = W * X  + B

w = 0.0
b = 0.0
lr = 0.01

def descent(x,y,w,b,lr):
  dldw = 0.0
  dldb = 0.0
  loss = 0.0
  N = x.shape[0]

  for xi , yi in zip(x,y):  
    loss += (yi - (w*xi + b))**2
    dldw += -2*xi*(yi-(w*xi+b))
    dldb += -2*(yi-(w*xi+b))

  w = w - lr * (1/N) * dldw
  b = b - lr * (1/N) * dldb
  loss = loss/N
  return w,b,loss

epoch = 700

for i in range(epoch):
  
    w, b , l = descent(X,Y,w,b,lr=lr)

    if i % 10 == 0:
       print(f"epoc: {i} | loss: {l.item():.4f} | Weight: {w.item():.4f} | bias: {b.item():.4f}")









