from sklearn.utils.extmath import randomized_svd
from sklearn.datasets import load_wine
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Carregar o conjunto de dados Wine
wine = load_wine()
dados = wine.data
# Tipos de vinhos, no exemplo foi utilizado para colorir os pontos do gráfico
cores = wine.target 

# Aplicar a Decomposição em Valores Singulares (SVD) 
U, S, VT = randomized_svd(dados, n_components=3)
# Reconstruir a matriz reduzida a partir das matrizes decompostas pelo SVD
resultados_svd = U.dot(np.diag(S))

# Extraia as três dimensões
x = resultados_svd[:, 0]
y = resultados_svd[:, 1]
z = resultados_svd[:, 2]

# Plote os dados em um gráfico 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(x, y, z, c=cores, cmap='viridis')
ax.set_xlabel('Dimensão 1')
ax.set_ylabel('Dimensão 2')
ax.set_zlabel('Dimensão 3')
plt.title('Visualização de Dados Reduzidos com SVD')
plt.show()