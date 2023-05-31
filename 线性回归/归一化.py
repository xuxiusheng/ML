import numpy as np
import math as mt
import matplotlib.pyplot as plt

m, n = 100, 2
np.random.seed(41)
X = 2 * np.random.randn(m, 1)
y = 4 + 3 * X + np.random.randn(m, 1)
X_b = np.c_[np.ones((m, 1)), X]

learning_rate = 0.0001
n_epochs = 4000


def loss_func(X, y, w):
    res = 0
    for i in range(X.shape[0]):
        y_pred = 0
        for j in range(X.shape[1]):
            y_pred += X[i, j] * w[j]
        res += pow(y_pred - y[i, 0], 2)
    return res


#未进行归一化
batch_size = 10
num_batch = mt.floor(m/batch_size)
unnormal_batch_theta = []
theta = np.random.randn(n, 1)
for epoch in range(n_epochs):
    arr = np.arange(len(X_b))
    np.random.shuffle(arr)
    X_b = X_b[arr]
    y = y[arr]
    for i in range(num_batch):
        x_batch = X_b[i * batch_size: i * batch_size + batch_size]
        y_batch = y[i * batch_size: i * batch_size + batch_size]
        gradient = x_batch.T.dot(x_batch.dot(theta) - y_batch)
        theta = theta - learning_rate * gradient
    unnormal_batch_theta.append(theta)
unnormal_batch_theta = np.array(unnormal_batch_theta).reshape(-1, n)
print(theta)

w1 = np.linspace(-2, 6, 2000)
w2 = np.linspace(-2, 5, 2000)
w1_, w2_ = np.meshgrid(w1, w2)
fig = plt.figure()
ax = plt.subplot(1, 1, 1)
ax.contour(w1_, w2_, loss_func(X_b, y, [w1_, w2_]), 100)
ax.plot(unnormal_batch_theta[:, 0], unnormal_batch_theta[:, 1], c='b', alpha=0.5,
        label='unnormalized')
ax.set_xlabel("w1")
ax.set_ylabel("w2")
ax.legend(loc="lower right")
plt.show()

#进行归一化
normal_batch_theta = []
X_b = (X_b - X_b.mean()) / X_b.std()
theta = np.random.randn(n, 1)
for epoch in range(n_epochs):
    arr = np.arange(len(X_b))
    np.random.shuffle(arr)
    X_b = X_b[arr]
    y = y[arr]
    for i in range(num_batch):
        x_batch = X_b[i * batch_size: i * batch_size + batch_size]
        y_batch = y[i * batch_size: i * batch_size + batch_size]
        gradient = x_batch.T.dot(x_batch.dot(theta) - y_batch)
        theta = theta - learning_rate * gradient
    normal_batch_theta.append(theta)
normal_batch_theta = np.array(normal_batch_theta).reshape(-1, n)
print(theta)


w1 = np.linspace(-2, 15, 2000)
w2 = np.linspace(-2, 8, 2000)
w1_, w2_ = np.meshgrid(w1, w2)
fig = plt.figure()
ax = plt.subplot(1, 1, 1)
ax.contour(w1_, w2_, loss_func(X_b, y, [w1_, w2_]), 100)
ax.plot(normal_batch_theta[:, 0], normal_batch_theta[:, 1], c='b', alpha=0.5,
        label='normalized')
ax.set_xlabel("w1")
ax.set_ylabel("w2")
ax.legend(loc="lower right")
plt.show()

