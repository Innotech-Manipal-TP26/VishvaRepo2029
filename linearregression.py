import numpy as np
import time


X = np.random.randn(10000, 1)
y = np.random.randn(10000)

lr = 0.01
epochs = 100
n = len(X)

w = 0.0
b = 0.0

start = time.time()

for _ in range(epochs):
    dw = 0.0
    db = 0.0

    for i in range(n):
        y_hat = w * X[i][0] + b
        error = y_hat - y[i]

        dw += error * X[i][0]
        db += error

    dw = (2 / n) * dw
    db = (2 / n) * db

    w = w - lr * dw
    b = b - lr * db

loop_time = time.time() - start

print(w, b)
print(loop_time)

w = 0.0
b = 0.0

start = time.time()

losses = []

for _ in range(epochs):
    y_hat = w * X[:, 0] + b
    error = y_hat - y

    loss = np.mean(error ** 2)
    losses.append(loss)

    dw = (2 / n) * np.sum(error * X[:, 0])
    db = (2 / n) * np.sum(error)

    w = w - lr * dw
    b = b - lr * db
vec_time = time.time() - start

print(w, b)
print(vec_time)
print(loop_time / vec_time)






print(losses[0], losses[-1])