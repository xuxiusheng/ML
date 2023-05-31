from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier


iris = load_iris()
X = iris.data
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                    random_state=42)
log_clf = LogisticRegression()
svc_clf = SVC()
dt_clf = DecisionTreeClassifier()

bag_clf = BaggingClassifier(log_clf, n_estimators=10, n_jobs=1, bootstrap=True)
bag_clf.fit(X_train, y_train)
y_hat_log = bag_clf.predict(X_test)
print(accuracy_score(y_test, y_hat_log))

voting_clf = VotingClassifier(estimators=[('lr', log_clf), ('dt', dt_clf),
                             ('svc', svc_clf)])
voting_clf.fit(X_train, y_train)
y_hat = voting_clf.predict(X_test)
print(accuracy_score(y_test, y_hat))