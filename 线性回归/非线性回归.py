import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error



m = 100

X = 6 * np.random.rand(m, 1)
y = 0.5 * X**2 + X + 2 + np.random.randn(m, 1)

fig = plt.figure()
ax = plt.subplot(1, 1, 1)
ax.patch.set_facecolor('silver')
ax.plot(X, y, 'bx', label="scatter data")


X_train = X[:80]
y_train = y[:80]
X_test = X[80:]
y_test = y[80:]

X_new = np.linspace(0, 6, 100).reshape(-1, 1)


d = {1: 'g-', 2: 'r+', 10: 'y*'}
for i in d:
    #经过转化之后已存在截距项
    poly_feature = PolynomialFeatures(degree=i, include_bias=True)
    X_poly_train = poly_feature.fit_transform(X_train)
    X_poly_test = poly_feature.fit_transform(X_test)

    linereg = LinearRegression(fit_intercept=False)#不再需要截距项
    linereg.fit(X_poly_train, y_train)

    X_new_transfer = poly_feature.fit_transform(X_new)
    y_pred = linereg.predict(X_new_transfer)
    ax.plot(X_new, y_pred, d[i], label='dimension' + str(i), )
plt.grid()
plt.legend()
plt.show()
