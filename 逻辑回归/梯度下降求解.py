from sklearn.datasets import load_breast_cancer
import numpy as np
import math as mt
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale

np.random.seed(41)
data = load_breast_cancer()
X, y = scale(data['data'][:, :2]), data['target'].reshape(-1, 1)
X = np.c_[np.ones((X.shape[0], 1)), X]
learning_rate = 0.0001
n_epoch = 1000


theta = np.random.randn(X.shape[1], 1)

#通过batch_size控制梯度策略
batch_size = 10
n_batches = mt.floor(X.shape[0] / batch_size)
for epoch in range(n_epoch):
    arr_index = np.arange(0, X.shape[0], 1)
    np.random.shuffle(arr_index)
    X = X[arr_index]
    y = y[arr_index]
    for i in range(n_batches):
        xi = X[i * batch_size:i * batch_size + batch_size]
        yi = y[i * batch_size:i * batch_size + batch_size]
        hx = 1 / (1 + np.exp(-xi.dot(theta)))
        gradient = xi.T.dot(hx - yi)
        theta = theta - learning_rate * gradient
theta = theta.ravel()
print(theta)

x_new = np.linspace(-2, 3, 1000)
y_new = -(theta[0] + theta[1] * x_new) / theta[2]
fig = plt.figure()
ax = plt.subplot(1, 1, 1)
ax.scatter(X[:, 1], X[:, 2], c=y)
ax.plot(x_new, y_new)
plt.show()