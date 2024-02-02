from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
import pandas as pd

# Dados de Ana e Carlos
filmes_de_ana = {'Filme': ['Ação', 'Comédia', 'Drama', 'Ação']}
vinhos_de_carlos = {'Vinho': ['Cabernet Sauvignon', 'Chardonnay', 'Merlot', 'Cabernet Sauvignon']}

# Criando DataFrames com os dados
df_ana = pd.DataFrame(filmes_de_ana)
df_carlos = pd.DataFrame(vinhos_de_carlos)

# Normalização e codificação de variáveis categóricas
label_encoder = LabelEncoder()
onehot_encoder = OneHotEncoder()

# Codificando preferências de Ana (filmes)
df_ana['Filme_LabelEncoded'] = label_encoder.fit_transform(df_ana['Filme'])
ana_encoded = onehot_encoder.fit_transform(df_ana[['Filme_LabelEncoded']]).toarray()

# Normalizando preferências de Carlos (vinhos)
df_carlos['Vinho_LabelEncoded'] = label_encoder.fit_transform(df_carlos['Vinho'])

# Exibindo os resultados
print("\nPreferências de Ana codificadas:")
print(ana_encoded)
print("\nPreferências de Carlos normalizadas:")
print(df_carlos)
print("")