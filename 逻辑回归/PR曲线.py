from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve
from sklearn.datasets import load_breast_cancer
import matplotlib.pyplot as plt


data = load_breast_cancer()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=66)
lr = LogisticRegression(solver='liblinear', max_iter=5000)
lr = lr.fit(X_train, y_train)
scores = lr.decision_function(X_test)
precisions, recalls, thresholds = precision_recall_curve(y_test, scores)
# x = np.linspace(0, 1, 10)
plt.plot(precisions, recalls, label='logistic')
# plt.plot(x, x)
plt.xlabel('recall')
plt.ylabel('precision')
plt.legend()
plt.show()

proba = lr.predict_proba(X_test)[:, 1]
fpr, tpr, threshold = roc_curve(y_test, proba)
plt.plot(fpr, tpr)
plt.plot([0, 1], [0, 1])
plt.show()
