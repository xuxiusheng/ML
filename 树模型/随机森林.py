from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris

iris = load_iris()
X = iris.data
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                    random_state=42)
clf = RandomForestClassifier(n_estimators=5, max_leaf_nodes=16, n_jobs=1,
                             oob_score=True)
clf.fit(X_train, y_train)
y_hat = clf.predict(X_test)
print(clf.oob_score_)
print(accuracy_score(y_test, y_hat))
for name, score in zip(iris.feature_names, clf.feature_importances_):
    print(name, score)
