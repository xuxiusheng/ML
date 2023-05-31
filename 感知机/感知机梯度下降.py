import numpy as np
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

X, y = make_blobs(n_samples=40, n_features=2,
                  centers=2, cluster_std=0.5,
                  shuffle=True, center_box=(-5, 5))

y[np.where(y == 0)[0]] = -1
print(y)
plt.scatter(X[:, 0], X[:, 1], c=y)


X = np.c_[np.ones((40, 1)), X]
theta = np.random.randn(3, 1)

error = 0
n_epochs = 100000
learning_rate = 0.1

for epoch in range(n_epochs):
    y_pre = X.dot(theta)
    ind = np.where((y * y_pre.ravel() <= 0) == True)[0]
    error = len(ind)
    if error == 0:
        print('存在解向量')
        break
    error_X, error_y = X[ind, :], -y[ind].reshape(-1, 1)
    gradient = np.sum((error_X * error_y), axis=0).reshape(-1, 1)
    theta = theta - learning_rate * gradient

theta = theta.ravel()

X_new = np.linspace(-3, 2, 100)
y_new = (-theta[0] - theta[1] * X_new) / theta[2]
plt.plot(X_new, y_new)
plt.show()
