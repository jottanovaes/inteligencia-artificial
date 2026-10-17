import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# custo de publicidade
# entrada, caracteristica, funcionalidades, etc.
X = [[10], [20], [30],[40],[50],[60],[70],[80],[90],[100]]

# vendas
# saida, alvo, etc
y = [[100], [200], [300],[400],[500],[600],[700],[800],[900],[1000]]

model = LinearRegression()

# treinamento do modelo
model.fit(X, y)

# previsão
print(model.predict([[155]]))
