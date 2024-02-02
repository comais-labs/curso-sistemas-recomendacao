from sklearn.utils.extmath import randomized_svd
import numpy as np

# Dados fictícios - Preferências de Ana, Carlos e mais 2 amigos por filmes e vinhos (0 a 5)
dados = np.array([
    [5, 4, 0, 0, 0],
    [0, 0, 3, 4, 5],
    [2, 4, 4, 3, 2],
    [0, 1, 5, 2, 0]
])

# Aplicar a Decomposição em Valores Singulares (SVD) 
U, S, VT = randomized_svd(dados, n_components=3)
# Reconstruir a matriz reduzida a partir das matrizes decompostas pelo SVD
dados_reduzidos = U.dot(np.diag(S))

print("\nSVD do SciKit Learn")
print("Dados originais:")
print(dados)
print("\nResultados SVD (3) componentes principais):")
print(np.round(dados_reduzidos, decimals=1))
print("")