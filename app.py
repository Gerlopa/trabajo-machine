from flask import Flask, render_template, request

from logistic import predecir, obtener_datos, obtener_roc_graph
from regression import calculate_regression, generate_graph_regression

app = Flask(__name__)


@app.route("/")
def home():
    return render_template("index.html")

@app .route("/quadratic_explanation")
def quadratic_explanation():
    return render_template("quadratic_explanation.html")

@app.route("/quadratic", methods=["GET", "POST"])
def quadratic_view():
    prediction = None

    # datos del modelo
    metrics = quadratic.datos_qda()
    roc_graph = quadratic.roc_qda()
    decision_graph = quadratic.grafica_decision_qda()  # 👈 FALTABA

    if request.method == "POST":
        try:
            columnas = metrics["columnas"]
            input_data = {col: float(request.form[col]) for col in columnas}
            prediction = quadratic.predecir_qda(input_data)
        except Exception as e:
            return f"Error en QDA: {e}"

    return render_template(
        "quadratic.html",
        prediction=prediction,
        metrics=metrics,
        roc_graph=roc_graph,
        decision_graph=decision_graph
    )


if __name__ == "__main__":
    app.run(debug=True)
