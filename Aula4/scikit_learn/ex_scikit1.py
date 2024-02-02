from sklearn.preprocessing import MinMaxScaler
import pandas as pd

# Dados de avaliações de filmes de Ana e avaliações de vinhos de Carlos
avaliacoes_de_ana = {'Avaliacoes': [5, 4, 3, 2, 5, 4, 3, 1, 2, 2, 3]}
avaliacoes_de_carlos = {'Avaliacoes': [90, 85, 70, 65, 95, 80, 75]}

# Criando DataFrames com as avaliações
df_ana = pd.DataFrame(avaliacoes_de_ana)
df_carlos = pd.DataFrame(avaliacoes_de_carlos)

# Normalização Min-Max para as avaliações de Ana
scaler_ana = MinMaxScaler()
df_ana['Avaliacoes_Normalizadas'] = scaler_ana.fit_transform(df_ana[['Avaliacoes']])

# Normalização Min-Max para as avaliações de Carlos
scaler_carlos = MinMaxScaler()
df_carlos['Avaliacoes_Normalizadas'] = scaler_carlos.fit_transform(df_carlos[['Avaliacoes']])

# Exibindo os resultados
print("Avaliações normalizadas de Ana:")
print(df_ana)
print("\nAvaliações normalizadas de Carlos:")
print(df_carlos)