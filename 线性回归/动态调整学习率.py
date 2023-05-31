import numpy as np
import math as mt
import matplotlib.pyplot as plt

#生成随机数种子
m, n = 100, 2
np.random.seed(41)
X = 2 * np.random.randn(m, 1)
y = 4 + 3 * X + np.random.randn(m, 1)
X_b = np.c_[np.ones((m, 1)), X]

n_epochs = 500


#动态修改学习率
t0, t1 = 5, 50000
def learning_rate_schedule(t):
    return t0/(t+t1)

batch_size = 10
num_batch = mt.floor(m/batch_size)
mini_batch_theta = []

learning_rates = []
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
        theta = theta - learning_rate_schedule(epoch) * gradient
    learning_rates.append(learning_rate_schedule(epoch))
    mini_batch_theta.append(theta)
mini_batch_theta = np.array(mini_batch_theta).reshape(-1, n)
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

fig = plt.figure(figsize=(20, 10))
ax = plt.subplot(1, 2, 1)
ax.contourf(w1_, w2_, loss_func(X_b, y, [w1_, w2_]), 50, cmap=plt.cm.hot)
C = ax.contour(w1_, w2_, loss_func(X_b, y, [w1_, w2_]), 50, cmap=plt.cm.hot)
plt.clabel(C, inline=True, fontsize=10)
ax.plot(mini_batch_theta[:, 0], mini_batch_theta[:, 1], c='b', alpha=0.5,
        label='mini-batch gradient', linewidth=5)


ax.set_xlabel("w1", size=15)
ax.set_ylabel("w2", size=15)
ax.legend(loc="lower right", borderpad=2)
ax.tick_params(labelsize=13)

ax = plt.subplot(1, 2, 2)
ax.patch.set_facecolor('silver')
ax.plot(range(n_epochs), learning_rates, label="learning_rate",
        linewidth=5)
ax.legend(borderpad=2)
plt.grid()

plt.tight_layout()
plt.show()