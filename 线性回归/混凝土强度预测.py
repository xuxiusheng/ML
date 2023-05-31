import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def load_data(path):
    data = pd.read_csv(path)
    print(data.shape)
    # print(data.isna().sum())

    low = np.round(data.quantile(0.25).tolist(), 2)
    up = np.round(data.quantile(0.75).tolist(), 2)

    gap = (up - low) * 1.5
    lower = low - gap
    upper = up + gap

    feature_names = data.columns.values
    for i in range(len(feature_names)):
        data[feature_names[i]] = data[feature_names[i]].apply(lambda x: x if x>=lower[i] and x<=upper[i] else np.nan)
    data.dropna(axis=0, inplace=True)
    data.index = range(len(data))

    fig = plt.figure()
    ax = plt.subplot(1, 1, 1)
    sns.heatmap(np.round(data.corr(), 2), annot=True)
    ax.set_title("相关系数")
    plt.show()

    sns.pairplot(data, diag_kind='kde')
    plt.show()

    sns.boxplot(y=data["Strength"].values.tolist())
    plt.show()

    relevance = data.corr()['Strength']
    new_features = []
    for name in feature_names[:-1]:
        if abs(relevance[name]) > 0.3:
            new_features.append(name)
    X = data.loc[:, new_features]
    y = data['Strength']

    return np.array(X), y.values

def model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    lr = LinearRegression().fit(X_train, y_train)
    y_pred = lr.predict(X_test)
    error = mean_squared_error(y_test, y_pred)
    print("训练集损失为:{}".format(round(error, 2)))
    print("训练集精度:{}".format(round(lr.score(X_train, y_train), 2)))
    print("验证集集精度:{}".format(round(lr.score(X_test, y_test), 2)))




if __name__ == "__main__":
    X, y = load_data('concrete.csv')
    model(X, y)