from sklearn.utils.extmath import randomized_svd
import numpy as np

# Matriz de exemplo
dados = np.array([
    [5, 3, 0, 1, 4],
    [4, 0, 0, 1, 3],
    [1, 1, 0, 5, 0],
    [1, 0, 0, 4, 4],
    [0, 1, 5, 4, 0]
])

# Aplicar a Decomposição em Valores Singulares (SVD) 
U, S, VT = randomized_svd(dados, n_components=5)

# Resultados
print("Matriz U:")
print(np.round(U, decimals=1))
print("\nMatriz Sigma:")
print(np.round(S, decimals=1))
print("\nMatriz VT:")
print(np.round(VT, decimals=1))