import numpy as np
from sklearn.utils.extmath import randomized_svd

# Matriz de preferências do usuário para itens
dados = np.array([
    [5, 3, 0, 1, 4],
    [4, 0, 0, 1, 3],
    [1, 1, 0, 5, 0],
    [1, 0, 0, 4, 4],
    [0, 1, 5, 4, 0]
])

# Aplicar a Decomposição em Valores Singulares (SVD) 
U, S, VT = randomized_svd(dados, n_components=2)

# Reconstruir a matriz a partir das matrizes decompostas pelo SVD
svd_reconstruida = np.dot(np.dot(U, np.diag(S)), VT)

# Arredondar os números da matriz reconstruída para 2 casa decimais após a vírgula
svd_reconstruida = np.round(svd_reconstruida, decimals=1)

print("Matriz Original:")
print(dados)
print("\nMatriz Reconstruída a partir dos Dados Reduzidos:")
print(svd_reconstruida)



# Prever classificações ausentes (supondo que queiramos prever as avaliações para o usuário 3)
usuario_a_prever = 1
print("\nClassificações originais do usuário:", dados[usuario_a_prever])
print("Classificações previstas para o usuário:", svd_reconstruida[usuario_a_prever])
print("")










'''
# Ajustar a escala dos valores reconstruídos para a escala original
min_original = np.min(dados)
max_original = np.max(dados)
min_reconstruido = np.min(svd_reconstruida)
max_reconstruido = np.max(svd_reconstruida)

# Escalar os valores reconstruídos para a escala original
svd_reconstruida = (
    (svd_reconstruida - min_reconstruido) / (max_reconstruido - min_reconstruido)
) * (max_original - min_original) + min_original
'''