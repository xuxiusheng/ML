import numpy as np
import math as mt
import matplotlib.pyplot as plt

#生成随机数种子
m, n = 100, 2
np.random.seed(41)
X = 2 * np.random.randn(m, 1)
y = 4 + 3 * X + np.random.randn(m, 1)
X_b = np.c_[np.ones((m, 1)), X]

learning_rate = 0.0001
n_epochs = 6000
alpha = 0.4



#批量梯度下降
batch_size = 10
num_batch = mt.floor(m/batch_size)
mini_batch_theta = []

theta = np.random.randn(n, 1)
for epoch in range(n_epochs):
    arr = np.arange(len(X_b))
    np.random.shuffle(arr)
    X_b = X_b[arr]
    y = y[arr]
    for i in range(num_batch):
        x_batch = X_b[i * batch_size: i * batch_size + batch_size]
        y_batch = y[i * batch_size: i * batch_size + batch_size]
        #lasso添加了L2正则项，求导时要对当前权重进行限制
        gradient = x_batch.T.dot(x_batch.dot(theta) - y_batch) + 2 * alpha * theta
        theta = theta - learning_rate * gradient
    mini_batch_theta.append(theta)
mini_batch_theta = np.array(mini_batch_theta).reshape(-1, n)
print(theta)


#全量梯度下降
batch_size = m
num_batch = mt.floor(m / batch_size)
batch_theta = []

theta = np.random.randn(n, 1)
for epoch in range(n_epochs):
    arr = np.arange(len(X_b))
    np.random.shuffle(arr)
    X_b = X_b[arr]
    y = y[arr]
    for i in range(num_batch):
        x_batch = X_b[i * batch_size: i * batch_size + batch_size]
        y_batch = y[i * batch_size: i * batch_size + batch_size]
        gradient = x_batch.T.dot(x_batch.dot(theta) - y_batch) + 2 * alpha * theta
        theta = theta - learning_rate * gradient
    batch_theta.append(theta)
batch_theta = np.array(batch_theta).reshape(-1, n)
print(theta)


#随机梯度下降
batch_size = 1
num_batch = mt.floor(m / batch_size)
stochastic_theta = []

theta = np.random.randn(n, 1)
for epoch in range(n_epochs):
    arr = np.arange(len(X_b))
    np.random.shuffle(arr)
    X_b = X_b[arr]
    y = y[arr]
    for i in range(num_batch):
        x_batch = X_b[i * batch_size: i * batch_size + batch_size]
        y_batch = y[i * batch_size: i * batch_size + batch_size]
        gradient = x_batch.T.dot(x_batch.dot(theta) - y_batch) + 2 * alpha *theta
        theta = theta - learning_rate * gradient
    stochastic_theta.append(theta)
stochastic_theta = np.array(stochastic_theta).reshape(-1, n)
print(theta)

def loss_func(X, y, w):
    res = 0
    for i in range(X.shape[0]):
        y_pred = 0
        for j in range(X.shape[1]):
            y_pred += X[i, j] * w[j]
        res += pow(y_pred - y[i, 0], 2)
    return res

w1 = np.linspace(-2, 6, 2000)
w2 = np.linspace(-2, 5, 2000)
w1_, w2_ = np.meshgrid(w1, w2)
fig = plt.figure()
ax = plt.subplot(1, 1, 1)
ax.contour(w1_, w2_, loss_func(X_b, y, [w1_, w2_]), 50)
ax.plot(batch_theta[:, 0], batch_theta[:, 1], c='r', alpha=0.5,
        label='batch gradient')
ax.plot(mini_batch_theta[:, 0], mini_batch_theta[:, 1], c='b', alpha=0.5,
        label='stochastic gradient')
ax.plot(stochastic_theta[:, 0], stochastic_theta[:, 1], c='g', alpha=0.5,
        label='mini-batch gradient')
ax.set_xlabel("w1")
ax.set_ylabel("w2")
ax.legend(loc="lower right")
plt.show()