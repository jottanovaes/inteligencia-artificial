
import pandas as pd
import joblib
from flask import Flask, render_template, request

# Etapa 4:  Integração com Aplicação Web em Flask

app = Flask(__name__)

# Carregando o modelo treinado
model = joblib.load("Modelos/modelo_veiculo.pkl")

@app.route("/")
def index():
    return render_template("index.html")

# Rota a previsão
@app.route("/resultado", methods=["POST"])
def resultado():
    potencia = float(request.form["potencia_cv"])

    # Criando o DataFrame para a previsão
    novo_veiculo = pd.DataFrame({"potencia_cv": [potencia]})
    # Predict do valor
    valor_estimado = model.predict(novo_veiculo)[0][0]

    # Formatando e apresentando o resusltado
    valor_formatado = f"R$ {valor_estimado:,.2f}"
    valor_formatado = valor_formatado.replace(",", "X").replace(".", ",").replace("X", ".")

    return render_template(
        "resultado.html",
        potencia=potencia,
        valor=valor_formatado
    )

if __name__ == "__main__":
    app.run(debug=True)
