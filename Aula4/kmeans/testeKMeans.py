import numpy as np
import pandas as pd
from scipy.sparse.linalg import svds
import matplotlib.pyplot as plt
from sklearn.utils.extmath import randomized_svd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore')

# Matriz de preferências do usuário para itens
movies = ['MIB', 'Star Trek', 'Ace Ventura', 'Coração Valente', 'Razão e Sensibilidade', 'Os Miseráveis']
users = ['Ana', 'Carlos', 'João', 'Maria', 'Joana', 'José', 'Enzo']

dados = pd.DataFrame([
    [5.0, 4.0, 0.0, 2.0, 2.0, 2.0],
    [4.0, 3.0, 4.0, 0.0, 3.0, 3.0],
    [5.0, 2.0, 5.0, 2.0, 1.0, 1.0],
    [3.0, 5.0, 3.0, 0.0, 1.0, 1.0],
    [3.0, 3.0, 3.0, 2.0, 4.0, 5.0],
    [2.0, 3.0, 2.0, 3.0, 5.0, 5.0],
    [2.0, 5.0, 0.0, 3.0, 3.0, 0.0]],
    columns=movies,
    index=users)

# Reduzir dimensionalidade para 2 dimensões com SVD
k = 2
U_filmes, S_filmes, VT_filmes = randomized_svd(dados.T, k)
U_usuario, S_usuario, VT_usuario = randomized_svd(dados, k)

# Reconstruir a matriz reduzida a partir das matrizes decompostas pelo SVD
X_truncated = U_filmes[:, :k] @ np.diag(S_filmes)[:k, :k]
X_truncated1 = U_usuario[:, :k] @ np.diag(S_usuario)[:k, :k]

# Concatenar os vetores de dimensão reduzidos dos usuarios e dos filmes em um única variável
concatenated = np.concatenate((X_truncated, X_truncated1), axis=0)
labels_usuarios_itens = np.concatenate((movies, users), axis=0)

# Plotas no gráfico os usuários e filmes com redução de dimensionalidade 
plt.scatter(concatenated[:, 0], concatenated[:, 1], cmap='viridis')
plt.xlabel('Dimensão 1')
plt.ylabel('Dimensão 2')
plt.title('Visualização de Dados Reduzidos')
for i, label in enumerate(labels_usuarios_itens):
    plt.annotate(label, (concatenated[i, 0], concatenated[i, 1]))
#plt.show()

# Laço para determinar a melhor quantidade de clusters
max=-1
qtCluster = -1
for i in range(2,10):
    # Aplicando o algoritmo K-Means
    kmeans = KMeans(n_clusters=i)
    kmeans.fit(concatenated)

    # Avaliar com o coeficiente de silhueta
    tmp = (silhouette_score(concatenated, kmeans.fit_predict(concatenated)))

    # Armazema o melhor resultado
    if tmp > max:
        max = tmp
        qtCluster = i

# Aplicando o algoritmo K-Means
kmeans = KMeans(n_clusters=qtCluster)
kmeans.fit(concatenated)

# Obtendo os centróides e rótulos dos clusters
centroids = kmeans.cluster_centers_
labels = kmeans.labels_

# Visualizando os clusters resultantes
plt.scatter(concatenated[:, 0], concatenated[:, 1], c=labels, s=50, cmap='viridis')
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='x', s=200, alpha=0.75, label='Centróides')
plt.title(f"Coeficiente de Silhueta médio: {max}\n")
for i, label in enumerate(labels_usuarios_itens):
    plt.annotate(label, (concatenated[i, 0], concatenated[i, 1]))
plt.show()