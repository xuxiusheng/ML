import numpy as np
import matplotlib.pyplot as plt

#生成随机数种子，每次生成固定的随机数
m = 100
np.random.seed(42)
X = np.random.rand(m, 1)
print(len(X))
print(X)


#生成对应的数据y，并添加高斯噪声
y = 4 * X + 5 + np.random.randn(100, 1)
print(y)

#第1列全部初始化为1，用于求解截距b
X_b = np.c_[np.ones((100, 1)), X]
print(X_b)

#w解析解形式
theta = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
print(theta)

X_new = np.array([[0], [2]])
X_new_b = np.c_[np.ones((2, 1)), X_new]
y_pre = X_new_b.dot(theta)
print(y_pre)

fig = plt.figure()
ax = plt.subplot(1, 1, 1)
ax.plot(X_new, y_pre, 'r-')
ax.plot(X, y, 'b.')
ax.axis([0, 1.2, 0, 15])
ax.set_xlabel("X")
ax.set_ylabel("y")
ax.set_title("Fitted curve")
ax.legend(labels=["y=" + str(np.round(theta[1, 0], 2)) +
                  "*x+" + str(np.round(theta[0, 0], 2)), "scatter data"], loc='lower right')
plt.show()