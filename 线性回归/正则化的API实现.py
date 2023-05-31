import numpy as np
from sklearn.linear_model import SGDRegressor


X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

#penalty为正则参数类型，可以为l1或l2以及elasticnet
#elasticnet的使用情况，不想让参数向量过于稀疏，又想达到正则化效果时，使用elasticnet
sgd_reg = SGDRegressor(penalty='l2', max_iter=1000)
sgd_reg = sgd_reg.fit(X, y.reshape(-1,))
print(sgd_reg.intercept_)
print(sgd_reg.coef_)


