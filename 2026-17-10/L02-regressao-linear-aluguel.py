import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib

# iniciando o dataset
df = pd.read_csv("/workspaces/inteligencia-artificial/2026-17-10/17 - metro-preco-aluguel.csv")

# separação entre características(X) e Target(y)
X = df[['Tamanho_m2']]
y = df['Preco']

# divisao em amostras 70/30
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,
    random_state=42
)

# treinamento do modelo
# instancia o modelo de regressão linear
modelo = LinearRegression()

# efetua o treinamento
modelo.fit(X_train, y_train)

# previsão
novo_imovel = pd.DataFrame({'Tamanho_m2': [330]})

previsao = modelo.predict(novo_imovel)
print(f"Preço estimado do aluguel: {previsao[0]:,.2f}")