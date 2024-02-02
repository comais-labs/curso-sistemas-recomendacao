from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

# Gerando dados de exemplo com Make Blobs, que é uma ferramenta útil
# para gerar conjuntos de dados de exemplo para testar algoritmos 
# de clustering ou classificação. No exemplo, ele é usado para 
# criar um conjunto de dados com 3 clusters distintos.
x, y = make_blobs(n_samples=100, centers=8, random_state=60, cluster_std=0.40)

# Visualizando os dados de exemplo gerados pelo Make Blobs
plt.scatter(x[:, 0], x[:, 1], s=50, cmap='viridis')
plt.title("Dados de Exemplo")
plt.show()

silhouette=[]
max=-1
qtCluster = -1
for i in range(2,15):
    # Aplicando o algoritmo K-Means
    kmeans = KMeans(n_clusters=i)
    kmeans.fit(x)

    # Avaliar com o coeficiente de silhueta
    tmp = (silhouette_score(x, kmeans.fit_predict(x)))

    # Armazema o melhor resultado
    if tmp > max:
        max = tmp
        qtCluster = i

    silhouette.append(tmp)
    
plt.plot(range(2,15), silhouette) 
# Visualizando os clusters resultantes
plt.title(f"Coeficiente de silhueta médio:\n")
plt.show()

# Aplicando o algoritmo K-Means
kmeans = KMeans(n_clusters=qtCluster)
kmeans.fit(x)

# Obtendo os centróides e rótulos dos clusters
centroids = kmeans.cluster_centers_
labels = kmeans.labels_

# Avaliar com o coeficiente de silhueta
silhouette_avg = silhouette_score(x, kmeans.fit_predict(x))

# Visualizando os clusters resultantes
plt.scatter(x[:, 0], x[:, 1], c=labels, s=50, cmap='viridis')
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='x', s=200, alpha=0.75, label='Centróides')
plt.title(f"Coeficiente de silhueta médio: {silhouette_avg}\n")
plt.legend()
plt.show()