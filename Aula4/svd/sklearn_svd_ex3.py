from sklearn.utils.extmath import randomized_svd
from sklearn.datasets import load_wine
import numpy as np
import matplotlib.pyplot as plt

# Carregar o conjunto de dados Wine
wine = load_wine()
dados = wine.data
# Tipos de vinhos, no exemplo foi utilizado para colorir os pontos do gráfico
cores = wine.target 

# Aplicar a Decomposição em Valores Singulares (SVD) 
U, S, VT = randomized_svd(dados, n_components=2)
# Reconstruir a matriz reduzida a partir das matrizes decompostas pelo SVD
resultados_svd = U.dot(np.diag(S))

# Extraia as 2 dimensões
x = resultados_svd[:, 0]
y = resultados_svd[:, 1]

# Plotar os resultados no plano cartesiado
plt.scatter(x, y, c=cores, cmap='viridis')
plt.xlabel('Dimensão 1')
plt.ylabel('Dimensão 2')
plt.title('Visualização de Dados Reduzidos com SVD')
plt.show()