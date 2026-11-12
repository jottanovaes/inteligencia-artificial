import pandas as pd
import math

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