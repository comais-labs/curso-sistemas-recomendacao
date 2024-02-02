from sklearn.utils.extmath import randomized_svd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Carregar o dataset de vinhos
file_path = 'dados_csv//xwines//XWines_Test_100_wines.csv'
data = pd.read_csv(file_path)

# Carregar o dataset de avaliações
ratings_file_path = 'dados_csv//xwines//XWines_Test_1K_ratings.csv'
ratings_data = pd.read_csv(ratings_file_path)

# Criar a matriz de avaliações
R_df = ratings_data.pivot(index='UserID', columns='WineID', values='Rating')
#print(R_df)

# Calcular a média das avaliações, descartando notas 0
medias = R_df.mean()

# Plotar o histograma das médias
plt.figure(figsize=(8, 6))
plt.hist(medias, bins=10, edgecolor='black')
plt.xlabel('Avaliação Média')
plt.ylabel('Frequência')
plt.title('Histograma das Avaliações Médias dos Vinhos')
plt.show()







'''
# Substituir NaN por 0
R = R_df.fillna(0).values
#print(R)

# Normalizar as avaliações subtraindo a média de cada usuário
user_ratings_mean = np.mean(R, axis=1)
R_demeaned = R - user_ratings_mean.reshape(-1, 1)
#print(R_demeaned)

# Plotar os resultados no plano cartesiado
# Aplicar a Decomposição em Valores Singulares (SVD) 
U, S, VT = randomized_svd(R.T, n_components=2)
# Reconstruir a matriz a partir dos dados reduzidos
resultados_svd = np.dot(np.dot(U, np.diag(S)), VT)

# Extraia as 2 dimensões
x = resultados_svd[:, 0]
y = resultados_svd[:, 1]

# Plotar os resultados no plano cartesiado
plt.scatter(x, y, cmap='viridis')
plt.xlabel('Dimensão 1')
plt.ylabel('Dimensão 2')
plt.title('Visualização de Dados Reduzidos com SVD')
plt.show()
'''