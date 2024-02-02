import numpy as np
import pandas as pd
from sklearn.utils.extmath import randomized_svd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Carregar o dataset de vinhos
file_path = 'dados_csv//xwines//XWines_Test_100_wines.csv'
data = pd.read_csv(file_path)

# Carregar o dataset de avaliações
ratings_file_path = 'dados_csv//xwines//XWines_Test_1K_ratings.csv'
ratings_data = pd.read_csv(ratings_file_path)

# Criar a matriz de avaliações
dados_originais = ratings_data.pivot(index='UserID', columns='WineID', values='Rating')
#print(dados_originais)

# Define todas as entradas iguais a NaN para 0
dados = dados_originais.fillna(0)

# Aplicar a Decomposição em Valores Singulares (SVD) 
k = 2
U_vinhos, S_vinhos, VT_vinhos = randomized_svd(dados.T, k)
U_usuarios, S_usuarios, VT_usuarios = randomized_svd(dados, k)

# Reconstruir a matriz reduzida a partir das matrizes decompostas pelo SVD
X_truncated = U_vinhos.dot(np.diag(S_vinhos))
X_truncated1 = U_usuarios.dot(np.diag(S_usuarios))

# Concatenar os vetores de dimensão reduzidos dos usuarios e dos filmes em um única variável
concatenated = np.concatenate((X_truncated, X_truncated1), axis=0)

# Obtendo o número de linhas da matriz X_truncated para criar os labels e atribuir 0s para ele
labels_vinhos = np.zeros(X_truncated.shape[0])
# Obtendo o número de linhas da matriz X_truncated1 para criar os labels e atribuir 1s para ele
labels_usuarios = np.ones(X_truncated1.shape[0])

# Plotas no gráfico os usuários e filmes com redução de dimensionalidade 
plt.scatter(X_truncated[:, 0], X_truncated[:, 1], c=labels_vinhos, cmap='viridis', s=22)
plt.scatter(X_truncated1[:, 0], X_truncated1[:, 1], c=labels_usuarios, cmap='viridis', s=22)
plt.colorbar()
plt.xlabel('Dimensão 1')
plt.ylabel('Dimensão 2')
plt.title('Visualização de Usuários e Vinhos do XWines\nRoxo = Vinho  |  Verde = Usuário')
#plt.show()

# Laço para determinar a melhor quantidade de clusters
max=-1
qtCluster = -1
for i in range(2,20):
    # Aplicando o algoritmo K-Means
    kmeans = KMeans(n_clusters=i)
    kmeans.fit(concatenated)

    # Avaliar com o coeficiente de silhueta
    tmp = (silhouette_score(concatenated, kmeans.fit_predict(concatenated)))

    # Armazema o melhor resultado em qtCluster
    if tmp > max:
        max = tmp
        qtCluster = i

# Aplicando o algoritmo K-Means
kmeans = KMeans(n_clusters=qtCluster)
kmeans.fit(concatenated)

# Obtendo os centróides e rótulos dos clusters
centroids = kmeans.cluster_centers_

# labes armazena os clusters de cada vinho e usuário
labels = kmeans.labels_

# Adicionar a coluna os clusters
concatenated = np.c_[concatenated, labels]

# Arredondar os números para duas casas decimais após a vírgula
concatenated = np.round(concatenated, decimals=2)

# Quebrar a matriz concatened em duas partes, sendo a primeira de vinhos e a segunda de usuários
vinhos = concatenated[:dados.shape[1]]
usuarios = concatenated[dados.shape[1]:]

# Criar DataFrame com os resultados
vinhos = pd.DataFrame(vinhos, index=data['WineID'], columns=['dimensao1', 'dimensao2', 'cluster'])
usuarios = pd.DataFrame(usuarios, index=dados_originais.index, columns=['dimensao1', 'dimensao2', 'cluster'])

# Configurar o Pandas para exibir todas as linhas
pd.set_option('display.max_rows', None)

# Imprimir a tabela de resultados dos vinhos
print("Vinhos:")
vinhos = vinhos.sort_values(by='cluster')
print(vinhos)

# Imprimir a tabela de resultados dos usuarios
print("\nUsuários:")
usuarios = usuarios.sort_values(by='cluster')
print(usuarios)

# Visualizando os clusters resultantes
plt.scatter(concatenated[:, 0], concatenated[:, 1], c=labels, s=50, cmap='viridis')
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='x', s=200, alpha=0.75, label='Centróides')
plt.title(f"Coeficiente de Silhueta médio: {max}\n")
#plt.show()