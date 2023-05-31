import numpy as np
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

np.random.seed(68)
X, y = make_blobs(n_samples=40, n_features=2,
                  centers=2, cluster_std=0.5,
                  shuffle=True, center_box=(-5, 5))

y[np.where(y == 0)[0]] = -1
print(y)
plt.scatter(X[:, 0], X[:, 1], c=y)

alpha = np.zeros((40, 1))
beta = 0
learning_rate = 0.001
n_epochs = 100000
for epoch in range(n_epochs):
    flag = False
    for i in range(X.shape[0]):
        y_pred = np.sum(alpha * y.reshape(-1, 1) * X, axis=0).dot(X[i, :])
        if y_pred * y[i] <= 0:
            alpha[i] = alpha[i] + learning_rate
            beta = beta + learning_rate * y[i]
            flag = True

    if flag == False:
        print("存在解析解")
        break
plt.show()