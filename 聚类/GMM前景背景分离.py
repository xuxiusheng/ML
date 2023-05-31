from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

img = Image.open('flower.jpg')
new_data = np.array(img).reshape(-1, 3)
gmm = GaussianMixture(n_components=2, covariance_type='tied').fit(new_data)
kmeans = KMeans(n_clusters=2).fit(new_data)

cluster_gmm = gmm.predict(new_data).reshape(800, 1200)
cluster_kmeans = kmeans.predict(new_data).reshape(800, 1200)

plt.figure(figsize=(16, 8), facecolor='w', edgecolor='k')
plt.subplot(1, 3, 1)
plt.title('GMM')
plt.imshow(cluster_gmm)
plt.subplot(1, 3, 2)
plt.imshow(img)
plt.title('source image')
plt.subplot(1, 3, 3)
plt.title('KMeans')
plt.imshow(cluster_kmeans)
plt.show()


