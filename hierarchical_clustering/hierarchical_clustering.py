from sklearn.datasets import make_blobs
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage

# Örnek veri oluşturma
X, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.6, random_state=42)

# Örnek veriyi görselleştirme
plt.figure()
plt.scatter(X[:, 0], X[:, 1])
plt.title('Ornek Veri')
plt.show()

# Bağlantı yöntemleri
linkage_methods = ['ward', 'single', 'average', 'complete']

# Her bir bağlantı yöntemi için kümeleme ve görselleştirme
plt.figure(figsize=(12, 8))
for i, method in enumerate(linkage_methods):
    model = AgglomerativeClustering(n_clusters=4, linkage=method)
    cluster_labels = model.fit_predict(X)
    
    # Dendrogram
    plt.subplot(2, 4, i+1)
    plt.title(f'{method.capitalize()} Linkage Dendogram')
    dendrogram(linkage(X, method=method), no_labels=True)
    plt.xlabel('Veri Noktalari')
    plt.ylabel('Uzaklik')
    
    # Kümeleme sonucu scatter plot
    plt.subplot(2, 4, i+5)
    plt.scatter(X[:, 0], X[:, 1], c=cluster_labels, cmap='viridis')
    plt.title(f'{method.capitalize()} Linkage Clustering')
    plt.xlabel('X')
    plt.ylabel('Y')

plt.tight_layout()
plt.show()
