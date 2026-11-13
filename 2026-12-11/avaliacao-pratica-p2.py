import os
import pandas as pd
import math
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib

## funcao para juntar lado a lado total e porcentagem de valores nulos
def resumo(df):
  df_conj_nulos = pd.concat([df.isnull().sum(), df.isnull().mean() * 100], axis=1, keys=["qtde", "porcentagem"])
  return df_conj_nulos

## carregando os datasets
clientes = pd.read_csv("clientes_veiculos.csv")
consultas = pd.read_csv("telemetria_consultas.csv")

## Etapa 1: Análise Exploratória & Limpeza de Dados (EDA)

## Diagnóstico e Imputação de Valores Ausentes:
## Calcule o total e a porcentagem de valores nulos em cada coluna.
print("Resumo de valores nulos no dataframe clientes:")
print(resumo(clientes))

## Aplique as regras de decisão para tratamento de ausentes:
## Remova linhas onde a chave única (cpf_cliente ou id_cliente) seja nula
## .copy() cria um DataFrame independente (evita o SettingWithCopyWarning ao imputar)
linhas_remover = clientes.dropna(subset=["cpf_cliente", "id_cliente"]).copy()

if linhas_remover.shape[0] < clientes.shape[0]:
  print(f"\n\nDF removido linha:\n {linhas_remover}")


## Remova a coluna que ultrapassa o limite de 50% de dados ausentes
## Informação muito incompleta para imputar
percentual_nulos_col = (linhas_remover.isna().sum() / len(linhas_remover)) * 100
colunas_remover = percentual_nulos_col[percentual_nulos_col > 50].index.tolist()

if colunas_remover:
  linhas_remover = linhas_remover.drop(columns=colunas_remover)
  print(f"\n\nDF removido coluna(s) {colunas_remover}:\n {linhas_remover}")
else:
  print("\n\nNenhuma coluna possui mais de 50% de dados ausentes. Nenhuma coluna removida.")


## Para variáveis numéricas contínuas com distribuição simétrica/normal
## (|skew| <= 0.5), impute os nulos com a Média.
## Identificação de Distribuição -> COEFICIENTE DE ASSIMETRIA (skew)
colunas_continuas = ["idade_cliente", "renda_mensal", "potencia_cv", "score_credito", "valor_avaliado_brt"]

for coluna in colunas_continuas:
  if linhas_remover[coluna].isna().sum() == 0:
    continue  # sem nulos, nada a imputar

  coef_assim = linhas_remover[coluna].skew()
  # Trunca o valor em uma casa decimal
  coef_assim = math.floor(coef_assim * 10) / 10
  # print(f"\n\n{coluna}: Coeficiente de Assimetria => {coef_assim}")

  if coef_assim >= -0.5 and coef_assim <= 0.5:
    # Obtém a MÉDIA e preenche os nulos com a MÉDIA
    media = linhas_remover[coluna].mean()
    linhas_remover[coluna] = linhas_remover[coluna].fillna(media)
    print(f"{coluna}: Distribuição simétrica => nulos imputados com a Média ({media:.2f})")
  else:
    ## Para variáveis com distribuição assimétrica (|skew| > 0.5) ou presença
    ## de outliers, impute os nulos com a Mediana.
    # Obtém a MEDIANA e preenche os nulos com a MEDIANA
    mediana = linhas_remover[coluna].median()
    linhas_remover[coluna] = linhas_remover[coluna].fillna(mediana)
    print(f"{coluna}: Distribuição assimétrica => nulos imputados com a Mediana ({mediana:.2f})")

print("\n\nResumo de valores nulos após a imputação:")
print(resumo(linhas_remover))

#==========================================================
#          IMPUTAÇÃO SEQUENCIAL E TEMPORAL
#==========================================================
## No dataframe de telemetria/série temporal, aplique ffill, bfill ou
## Interpolação Linear (interpolate()) conforme a melhor lógica observada.
print("\n\nResumo de valores nulos no dataframe consultas:")
print(resumo(consultas))

# Garante a ordem cronológica antes de imputar
consultas["data_hora"] = pd.to_datetime(consultas["data_hora"])
consultas = consultas.sort_values("data_hora")

print("\n\nDataFrame Antes:")
print(consultas)

# Comparação dos 3 métodos para observar a melhor lógica
df_temp = consultas.copy()
df_temp["ffill"] = consultas["requisicoes_minuto"].ffill()
df_temp["bfill"] = consultas["requisicoes_minuto"].bfill()
df_temp["interpolate"] = consultas["requisicoes_minuto"].interpolate()
print(f"\n\nComparação ffill x bfill x interpolate:\n{df_temp[consultas['requisicoes_minuto'].isna()]}")

# Decisão: Interpolação Linear
# - A série tem tendência crescente constante (120 -> 185) em intervalos regulares de 10 min.
# - ffill repete o valor anterior (ex: 128, 128 e depois salta para 140) -> cria "degraus" falsos.
# - bfill antecipa o valor futuro (ex: 140, 140 após 128) -> também cria degraus.
# - interpolate() traça uma reta entre o ponto anterior e o posterior (ex: 132, 136),
#   respeitando a tendência. Os nulos estão todos no meio da série, então não sobram NaN.
consultas["requisicoes_minuto"] = consultas["requisicoes_minuto"].interpolate()

print(f"\n\nInterpolação Linear:\n{consultas}")

print("\n\nResumo de valores nulos após a imputação (consultas):")
print(resumo(consultas))

# Etapa 1: Tratamento de Registros Duplicados

# Identificando duplicatas exatas
print("\nQuantidade de duplicatas exatas:")
print(linhas_remover.duplicated().sum())

# Removendo duplicatas exatas
df_sem_duplicadas = linhas_remover.drop_duplicates().copy()

print("\nAntes da remoção:", len(linhas_remover))
print("\nDepois da remoção:", len(df_sem_duplicadas))

# Identificando duplicatas parciais pelo CPF

# Convertendo data_solicitacao para datetime
df_sem_duplicadas['data_solicitacao'] = pd.to_datetime(df_sem_duplicadas['data_solicitacao'])

# Ordenando por CPF e data de solicitação
df_ordenado = df_sem_duplicadas.sort_values(
    by=['cpf_cliente', 'data_solicitacao'],
    ascending=[True, True]
)

# Removendo duplicatas parciais, mantendo a última ocorrência
df_final = df_ordenado.drop_duplicates(
    subset=['cpf_cliente'],
    keep='last'
).reset_index(drop=True)

print("\nDataFrame após remoção das duplicatas parciais:")
print(df_final)

print("\nQuantidade de registros após tratamento:")
print(len(df_final))

# atualizando dataFrame 
linhas_remover = df_final.copy()

# Detecção de Outliers
col_numericas = [
    'idade_cliente',
    'renda_mensal',
    'potencia_cv',
    'ano_fabricacao',
    'score_credito',
    'valor_avaliado_brt'
]

# boxplots
fig, ax = plt.subplots(2, 4, figsize=(12, 6))

axes_achatados = ax.flatten()

for i, nome_col in enumerate(col_numericas):
    axes_achatados[i].set_title(nome_col)

    sns.boxplot(
        y=linhas_remover[nome_col],
        ax=axes_achatados[i]
    )

for j in range(len(col_numericas), len(axes_achatados)):
    fig.delaxes(axes_achatados[j])

plt.tight_layout()
plt.show()

# kdeplots
fig, ax = plt.subplots(2, 4, figsize=(12, 6))

axes_achatados = ax.flatten()

for i, nome_col in enumerate(col_numericas):
    axes_achatados[i].set_title(nome_col)

    sns.kdeplot(
        x=linhas_remover[nome_col],
        ax=axes_achatados[i]
    )

for j in range(len(col_numericas), len(axes_achatados)):
    fig.delaxes(axes_achatados[j])

plt.tight_layout()
plt.show()


# regra do IQR
print("\n\nCalculando os limites pelo IQR")

for nome_col in col_numericas:

    Q1 = linhas_remover[nome_col].quantile(0.25)
    Q3 = linhas_remover[nome_col].quantile(0.75)

    IQR = Q3 - Q1

    limite_inferior = Q1 - 1.5 * IQR
    limite_superior = Q3 + 1.5 * IQR

    print(f"\nColuna: {nome_col}")
    print(f"Q1: {Q1:.2f}")
    print(f"Q3: {Q3:.2f}")
    print(f"IQR: {IQR:.2f}")
    print(f"Limite inferior: {limite_inferior:.2f}")
    print(f"Limite superior: {limite_superior:.2f}")


# aplicando CAPPING (WINSORIZAÇÃO)
print("\n\nAplicando Capping nos valores extremos")

for nome_col in col_numericas:

    Q1 = linhas_remover[nome_col].quantile(0.25)
    Q3 = linhas_remover[nome_col].quantile(0.75)

    IQR = Q3 - Q1

    limite_inferior = Q1 - 1.5 * IQR
    limite_superior = Q3 + 1.5 * IQR

    # Limitando valores inferiores
    linhas_remover[nome_col] = np.where(
        linhas_remover[nome_col] < limite_inferior,
        limite_inferior,
        linhas_remover[nome_col]
    )

    # Limitando valores superiores
    linhas_remover[nome_col] = np.where(
        linhas_remover[nome_col] > limite_superior,
        limite_superior,
        linhas_remover[nome_col]
    )

    print(f"\nCapping aplicado: {nome_col}")
    print(f"Limite inferior: {limite_inferior:.2f}")
    print(f"Limite superior: {limite_superior:.2f}")

print("\nDataFrame após aplicação do Capping:")
print(linhas_remover)


# Etapa2: Divisão da amostra e treinamento modelo

# Definindo as variáveis X e y
X = linhas_remover[['potencia_cv']]
y = linhas_remover[['valor_avaliado_brt']]

# Dividindo os dados em: 70% treinamento e 30% teste
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Exibindo a quantidade de registros em cada conjunto
print("\n\nETAPA 2: Divisão da Amostra")

print("\nQuantidade de registros:")
print(f"Treinamento: {len(X_train)} \t Teste: {len(X_test)}")
print(f"\nDimensões: X_train: {X_train.shape}, X_test: {X_test.shape}, y_train: {y_train.shape}, y_teste: {y_test.shape}")

# Criando o modelo de Regressão Linear
model = LinearRegression()

# Treinando o modelo
model.fit(X_train, y_train)

print("\nModelo de Regressão Linear treinado com sucesso!")

# Etapa 3: Exportação do Modelo 

# Criando o diretório Modelos
os.makedirs("Modelos", exist_ok=True)

# Salvando o modelo treinado
joblib.dump(model, "Modelos/modelo_veiculo.pkl")

print("\nModelo salvo!")