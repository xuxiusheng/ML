from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import scale

data = load_breast_cancer()
X, y = scale(data['data'][:, :2]), data['target']



lr = LogisticRegression(fit_intercept=True)
lr.fit(X, y)

theta = lr.coef_[0]
theta1 = np.linspace(theta[0] - 2, theta[0] + 2)
theta2 = np.linspace(theta[1] - 2, theta[1] + 2)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def loss_func(X, y, theta):
    m, n = X.shape
    loss = 0
    for i in range(m):
        y_pred = 0
        for j in range(n):
            y_pred += X[i, j] * theta[j]
        p = sigmoid(y_pred)
        cross_entropy = -y[i] * np.log(p) - (1 - y[i]) * np.log(1 - p)
        loss += cross_entropy
    return loss


w1, w2 = np.meshgrid(theta1, theta2)
fig = plt.figure(figsize=(10, 15))
ax = plt.subplot(2, 1, 1)
ax.contour(w1, w2, loss_func(X, y, [w1, w2]), 50, cmap=plt.cm.hot)
ax = plt.subplot(2, 1, 2)
ax.scatter(X[:, 0], X[:, 1], c=y)
x_new = np.linspace(-2, 2, 100)
y_new = -(lr.intercept_ + theta[0] * x_new) / theta[1]
ax.plot(x_new, y_new)
plt.show()



