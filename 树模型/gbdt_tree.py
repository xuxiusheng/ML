from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import roc_auc_score

class GradientBoostingWithLR:
    def __init__(self):
        self.gbdt_model = None
        self.lr_model = None
        self.gbdt_encoder = None
        self.X_train_leafs = None
        self.X_test_leafs = None
        self.X_trans = None

    def gbdt_train(self, x, y):
        gbdt_model = GradientBoostingClassifier(n_estimators=5, max_depth=2, max_features=0.5)
        gbdt_model.fit(x, y)
        return gbdt_model

    def lr_train(self, x, y):
        lr_model = LogisticRegression()
        lr_model.fit(x, y)
        return lr_model

    def gbdt_lr_train(self, x, y):
        self.gbdt_model = self.gbdt_train(x, y)
        self.X_train_leafs = self.gbdt_model.apply(x)[:, :, 0]
        self.gbdt_encoder = OneHotEncoder(categories='auto', sparse=False)
        self.X_trans = self.gbdt_encoder.fit_transform(self.X_train_leafs)
        self.lr_model = self.lr_train(self.X_trans, y)
        return self.lr_model

    def gbdt_lr_pred(self, model, x, y):
        self.X_test_leafs = self.gbdt_model.apply(x)[:, :, 0]
        x_trans = self.gbdt_encoder.transform(self.X_test_leafs)
        y_pred = model.predict_proba(x_trans)[:, 1]
        auc_score = roc_auc_score(y, y_pred)
        print("GBDT+LR AUC score: %.5f" % auc_score)
        return auc_score

    def model_assessment(self, model, x, y):
        y_pred = model.predict_proba(x)[:, 1]
        auc_score = roc_auc_score(y, y_pred)
        print("GBDT AUC score: %.5f" % auc_score)
        return auc_score

def loadData():
    iris = load_iris()
    X, y = iris.data, (iris.target == 2).astype(int)
    return train_test_split(X, y, test_size=0.3, random_state=0)


if __name__ == '__main__':
    X_train, X_test, y_train, y_test = loadData()
    print(X_train)
    print(y_train)
    model = GradientBoostingWithLR()
    lr = model.gbdt_lr_train(X_train, y_train)
    model.gbdt_lr_pred(lr, X_test, y_test)
    model.model_assessment(model.gbdt_model, X_test, y_test)
    pass


