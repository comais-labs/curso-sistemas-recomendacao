from sklearn.datasets import load_wine
import matplotlib.pyplot as plt
import pandas as pd

# Carregando o dataset Wine do scikit-learn
wine = load_wine()

# Convertendo o dataset em um DataFrame pandas para facilitar a visualização
dados = pd.DataFrame(wine.data, columns=wine.feature_names)
#dados['target'] = wine.target

# Explorando características dos vinhos - Ana
print("Características dos vinhos:")
print(dados.head())

# Visualizando a distribuição de algumas características - Carlos
plt.figure(figsize=(12, 6))

plt.subplot(2, 2, 1)
plt.hist(dados['alcohol'], bins=20, color='skyblue', edgecolor='black')
plt.title('Distribuição do Teor Alcoólico')
plt.xlabel('Teor Alcoólico')
plt.ylabel('Frequência')

plt.subplot(2, 2, 2)
plt.scatter(dados['flavanoids'], dados['color_intensity'], cmap='viridis', alpha=0.7)
plt.title('Flavonoides vs Intensidade de Cor')
plt.xlabel('Flavonoides')
plt.ylabel('Intensidade de Cor')

plt.tight_layout()
plt.show()