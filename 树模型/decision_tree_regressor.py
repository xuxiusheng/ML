from sklearn.tree import DecisionTreeRegressor
import numpy as np
import matplotlib.pyplot as plt

N = 100
X = np.random.rand(N) * 6 - 3
X.sort()

y = np.sin(X) + np.random.rand(N) * 0.05

X = X.reshape(-1, 1)
clf = DecisionTreeRegressor(criterion='mse', max_depth=3)
clf.fit(X, y)

X_test = np.linspace(-3, 3, 50).reshape(-1, 1)
y_hat = clf.predict(X_test)

plt.figure()
plt.plot(X, y, 'y*', label='true')
plt.plot(X_test, y_hat, 'b-', linewidth=2, label='prediction')
plt.legend()
plt.show()

depth = [2, 4, 6, 8, 10]
color = 'rgbmy'
plt.plot(X, y, 'k*', label='true')
for d, c in zip(depth, color):
    clf = DecisionTreeRegressor(max_depth=d, criterion='mse')
    clf.fit(X, y)
    y_hat = clf.predict(X_test)
    plt.plot(X_test, y_hat, '-', color=c, linewidth=2, label='depth=%d' %d)
plt.legend()
plt.show()
