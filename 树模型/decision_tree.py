import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import matplotlib as mpl

iris = load_iris()
data = pd.DataFrame(iris.data, columns=iris.feature_names)
data['Species'] = iris.target
# print(data.head(5))

X = data.iloc[:, 2:4]
y = data['Species']

# print(X.head(5))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

clf = DecisionTreeClassifier(max_depth=8, criterion='gini')
clf = clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print("Accuracy is ", accuracy_score(y_test, y_pred))

print(clf.feature_importances_)

export_graphviz(clf, out_file='./iris_decision_tree.dot',
                feature_names=iris.feature_names[2:4],
                class_names=iris.target_names,
                rounded=True,
                filled=True)

