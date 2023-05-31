import sklearn.datasets as ds
import matplotlib
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
import numpy as np

if __name__ == '__main__':
    N = 1000
    centers = [[1, 2], [-1, -1], [1, -1], [-1, 1]]
    data, y = ds.make_blobs(N, n_features=2, centers=centers, cluster_std=[0.5, 0.25, 0.7, 0.5],
                            random_state=2)
    params = ([0.2, 5], [0.2, 10], [0.2, 15], [0.3, 5], [0.3, 10], [0.3, 15])
    matplotlib.rcParams['font.sans-serif'] = [u'SimHei']
    matplotlib.rcParams['axes.unicode_minus'] = False

    plt.figure(figsize=(12, 8), facecolor='w')
    plt.suptitle('DBSCAN聚类', fontsize=20)

    for i in range(6):
        eps, min_samples = params[i]
        model = DBSCAN(eps=eps, min_samples=min_samples).fit(data)
        y_hat = model.labels_

        core_indices = np.zeros((len(y_hat), ), dtype=bool)
        core_indices[model.core_sample_indices_] = True

        y_unique = np.unique(y_hat)
        n_clusters = len(y_unique) - (1 if -1 in y_hat else 0)

        plt.subplot(2, 3, i + 1)
        clrs = plt.cm.Spectral(np.linspace(0, 0.8, y_unique.size).reshape(1, -1))
        for k, clr in zip(y_unique, clrs[0]):
            cur = (y_hat == k)
            if k == -1:
                plt.scatter(data[cur, 0], data[cur, 1], s=20, c='k')
                continue
            plt.scatter(data[cur, 0], data[cur, 1], s=30, c=clr, edgecolors='k')
            plt.scatter(data[cur & core_indices, 0], data[cur & core_indices, 1], s=60, c=clr,
                        marker='X')
            plt.grid(True)
            plt.title(f'epsilon = {eps}, min_samples={min_samples}', fontsize=16)
    plt.tight_layout()
    plt.show()