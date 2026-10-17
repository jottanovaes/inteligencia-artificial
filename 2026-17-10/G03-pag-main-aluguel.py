from flask import Flask, render_template, request, url_for
import joblib
import pandas as pd

# criar o servidor com flask
app = Flask(__name__)

# carrega o modelo# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Carrega o modelo de Machine Learning apenas UMA VEZ na inicialização
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
try:
    modelo = joblib.load('./2026-17-10/modelos/modelo_aluguel.pkl')
    print("\n---------------------------------------------------")
    print("Modelo 'modelo_aluguel.pkl' carregado com sucesso!")
    print("---------------------------------------------------\n")
except Exception as e:
    print("\n---------------------------------------------------")
    print(f"Erro ao carregar o modelo: {e}")
    print("---------------------------------------------------\n")
    modelo = None

# cria rotas
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/resultado', methods=['POST'])
def resultado():
    # Obtém o valor digitado no formulário
    tamanho_m2 = request.form.get('tamanho_m2')
    
    # Validação da entrada
    if not tamanho_m2:
        return render_template("resultado.html", error="Nenhum valor informado.")
    
    try:
        tamanho_m2 = int(tamanho_m2)
    except ValueError:
        return render_template("resultado.html", error="Por favor, insira um número inteiro válido.")
    
    # Faz a previsão se o modelo estiver carregado
    if modelo is None:
        return render_template("resultado.html", error="Modelo de Machine Learning não está disponível.")

    df_X = pd.DataFrame({'Tamanho_m2': [tamanho_m2]})

    # Executa a previsão
    predicao = modelo.predict(df_X)

    # Arredonda para 2 casas decimais
    valor_previsto = round(float(predicao[0]), 2)

    # Envia o tamanho e o valor previsto para o template resultado.html
    return render_template(
        "resultado.html", 
        tamanho_m2=tamanho_m2, 
        valor_previsto=valor_previsto
    )
# executa a aplicação, precisa ser o ultimo bloco
if __name__ == '__main__':    
    app.run(debug=True)