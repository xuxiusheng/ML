import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors
from sklearn.cluster import spectral_clustering
from sklearn.metrics import euclidean_distances

if __name__ == '__main__':
    matplotlib.rcParams['font.sans-serif'] = [u'SimHei']
    matplotlib.rcParams['axes.unicode_minus'] = False

    t = np.arange(0, 2 * np.pi, 0.1)
    data1 = np.vstack((np.cos(t), np.sin(t))).T
    data2 = np.vstack((2 * np.cos(t), 2 * np.sin(t))).T
    data3 = np.vstack((3 * np.cos(t), 3 * np.sin(t))).T
    data = np.vstack((data1, data2, data3))

    n_clusters = 3
    m = euclidean_distances(data, squared=True)
    sigma = np.median(m)

    plt.figure(figsize=(12, 8), facecolor='w')
    plt.suptitle('谱聚类', fontsize=20)
    clrs = plt.cm.Spectral(np.linspace(0, 0.8, n_clusters))

    for i, s in enumerate(np.logspace(-2, 0, 6)):
        print(s)
        af = np.exp(-m ** 2 / (s ** 2)) + 1e-6
        y_hat = spectral_clustering(af, n_clusters=n_clusters, assign_labels='kmeans',
                                    random_state=2)
        plt.subplot(2, 3, i+1)
        for k, clr in enumerate(clrs):
            cur = (y_hat == k)
            plt.scatter(data[cur, 0], data[cur, 1], s=40, c=clr, edgecolors='k')
        plt.title(f'sigma={round(s, 2)}')
    plt.show()
