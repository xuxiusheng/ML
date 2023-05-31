import numpy as np
import matplotlib.pyplot as plt
import sklearn.datasets as ds
import matplotlib.colors
from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans
from sklearn.mixture import GaussianMixture

def expand(a, b):
    d = (b - a) * 0.1
    return a - d, b + d

if __name__ == '__main__':
    N = 400
    centers = 4
    data, y = ds.make_blobs(N, n_features=2, centers=centers, random_state=2)
    data2, y2 = ds.make_blobs(N, n_features=2, centers=centers, cluster_std=(1, 2.5, 0.5, 2),
                              random_state=2)
    data3 = np.vstack((data[y == 0][:, :], data[y == 1][:50, :], data[y == 2][:20, :],
                       data[y == 3][:5, :]))
    y3 = np.array([0] * 100 + [1] * 50 + [2] * 20 + [3] * 5)

    cls = KMeans(n_clusters=4, init='k-means++')
    y_hat = cls.fit_predict(data)
    y2_hat = cls.fit_predict(data2)
    y3_hat = cls.fit_predict(data3)

    m = np.array([[1, 1], [1, 3]])
    data_r = data.dot(m)
    y_r_hat = cls.fit_predict(data_r)
    matplotlib.rcParams['font.sans-serif'] = [u'SimHei']
    matplotlib.rcParams['axes.unicode_minus'] = False
    cm = matplotlib.colors.ListedColormap(list('rgbm'))

    plt.figure(figsize=(9, 10), facecolor='w')
    plt.subplot(421)
    plt.title('原始数据')
    plt.scatter(data[:, 0], data[:, 1], c=y, s=30, cmap=cm, edgecolors='none')
    plt.grid(True)

    plt.subplot(422)
    plt.title("KMeans++聚类")
    plt.scatter(data[:, 0], data[:, 1], c=y_hat, cmap=cm, edgecolors='none')
    plt.grid(True)

    plt.subplot(423)
    plt.title('旋转后数据')
    plt.scatter(data_r[:, 0], data_r[:, 1], c=y, s=30, cmap=cm, edgecolors='none')
    plt.grid(True)

    plt.subplot(424)
    plt.title('旋转后KMeans++聚类')
    plt.scatter(data_r[:, 0], data_r[:, 1], c=y_r_hat, cmap=cm, edgecolors='none')
    plt.grid(True)

    plt.subplot(425)
    plt.title("方差不相等数据")
    plt.scatter(data2[:, 0], data2[:, 1], c=y2, s=30, cmap=cm, edgecolors='none')
    plt.grid(True)

    plt.subplot(426)
    plt.title('方差不相等KMeans++聚类')
    plt.scatter(data2[:, 0], data2[:, 1], c=y2_hat, cmap=cm, edgecolors='none')
    plt.grid(True)

    plt.subplot(427)
    plt.title('数量不相等聚类')
    plt.scatter(data3[:, 0], data3[:, 1], c=y3, cmap=cm, s=30, edgecolors='none')
    plt.grid(True)

    plt.subplot(428)
    plt.title('数量不相等KMeans++聚类')
    plt.scatter(data3[:, 0], data3[:, 1], c=y3_hat, cmap=cm, edgecolors='none')
    plt.grid(True)
    plt.show()