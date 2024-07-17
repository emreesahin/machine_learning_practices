from sklearn import datasets, cluster
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Veri kümelerini oluşturma
n_samples = 1500
noisy_circles = datasets.make_circles(n_samples=n_samples, factor=0.5, noise=0.05)
noisy_moons = datasets.make_moons(n_samples=n_samples, noise=0.05)
blobs = datasets.make_blobs(n_samples=n_samples)
no_structure = (np.random.rand(n_samples, 2), None)

datasets = [noisy_circles, noisy_moons, blobs, no_structure]

clustering_names = ['MiniBatchKMeans', 'SpectralClustering', 'Ward', 'AgglomerativeClustering', 'DBSCAN', 'Birch']

# Renk paleti
colors = np.array(['b', 'g', 'r', 'c', 'm', 'y'])

plt.figure(figsize=(len(clustering_names) * 3, len(datasets) * 3))

for i_dataset, dataset in enumerate(datasets):
    X, y = dataset
    X = StandardScaler().fit_transform(X)
    
    # Kümeleme algoritmalarını tanımlama
    clustering_algorithms = [
        cluster.MiniBatchKMeans(n_clusters=2),
        cluster.SpectralClustering(n_clusters=2, affinity='nearest_neighbors'),
        cluster.AgglomerativeClustering(n_clusters=2, linkage='ward'),
        cluster.AgglomerativeClustering(n_clusters=2, linkage='average'),
        cluster.DBSCAN(eps=0.2),
        cluster.Birch(n_clusters=2)
    ]

    for i_algo, (name, algo) in enumerate(zip(clustering_names, clustering_algorithms)):
        algo.fit(X)
        
        if hasattr(algo, 'labels_'):
            y_pred = algo.labels_.astype(int)
        else:
            y_pred = algo.predict(X)

        plt.subplot(len(datasets), len(clustering_algorithms), i_dataset * len(clustering_algorithms) + i_algo + 1)
        
        if i_dataset == 0:
            plt.title(name, size=12)
        
        plt.scatter(X[:, 0], X[:, 1], c=colors[y_pred].tolist(), s=10)
        plt.xticks(())
        plt.yticks(())

plt.tight_layout()
plt.show()
