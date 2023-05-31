import numpy as np
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import math as mt
import scipy

np.random.seed(4)
X, y = make_blobs(n_samples=300, n_features=2,
                  centers=2, cluster_std=0.5,
                  shuffle=True, center_box=(-5, 5))

y = y.reshape(-1, 1)
# print(y)
plt.figure(num=None, facecolor='w', edgecolor='k')
plt.scatter(X[:, 0], X[:, 1], c=y)
# plt.show()

X_b = np.c_[np.ones((X.shape[0], 1)), X]

theta = np.random.randn(X_b.shape[1], 1)
# print(theta.shape)

m, n = X_b.shape[0], X_b.shape[1]
batch_size = X_b.shape[0]
n_batch = mt.floor(X_b.shape[0] / batch_size)
eps = 0.1
flag = False
learning_rate = 1
t = 0

def sigmoid(x):
    if x >= 0:
        return 1.0 / (1 + np.exp(- x))
    else:
        return np.exp(x) * 1.0 / (1 + np.exp(x))
sigmoid = np.frompyfunc(sigmoid, 1, 1)

def hessian(theta, xi, m):
    hx = sigmoid(xi.dot(theta)).astype(np.float)
    p = hx * (1 - hx)
    W = np.diag(p[:, 0])
    hessian_matrix = xi.T.dot(W).dot(xi)

    return scipy.linalg.inv(hessian_matrix)

while t < 5:
    arr = np.arange(0, X_b.shape[0], 1)
    np.random.shuffle(arr)
    X_b = X_b[arr]
    y = y[arr]
    for i in range(n_batch):
        xi = X_b[i * batch_size: i * batch_size + batch_size]
        yi = y[i * batch_size: i * batch_size + batch_size]
        hx = sigmoid(xi.dot(theta))
        gradient = xi.T.dot(hx - yi)
        hessian_matrix = hessian(theta, X_b, batch_size)
        old_theta = theta
        theta = theta - hessian_matrix.dot(gradient)
    # print(theta)
    if (np.abs(old_theta - theta) < eps).sum() == n:
        t += 1
    else:
        t = 0

theta = theta.astype(np.float)
theta = np.round(theta.ravel(), 4)
print(theta)

x = np.linspace(-6, 4, 1000)
new_y = (-theta[0] - theta[1] * x) / theta[2]

plt.plot(x, new_y, 'r-')
plt.show()

