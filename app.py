from flask import Flask, render_template, request
from logistic import predecir, obtener_datos, obtener_roc_graph
import quadratic
from regression import calculate_regression, generate_graph_regression, obtener_metricas

app = Flask(__name__)


@app.route("/")
def home():
    return render_template("index.html")
@app.route("/case1")
def case1():
    return render_template("case1.html")

@app.route("/case2")
def case2():
    return render_template("case2.html")

@app.route("/case3")
def case3():
    return render_template("case3.html")

@app.route("/case4")
def case4():
    return render_template("case4.html")


@app.route("/linear_explanation")
def linear_explanation():
    return render_template("linear_explanation.html")

@app.route("/logistic_explanation")
def logistic_explanation():
    return render_template("logistic_explanation.html")

@app .route("/quadratic_explanation")
def quadratic_explanation():
    return render_template("quadratic_explanation.html")

@app.route("/regression", methods=["GET", "POST"])
def regression_view():
    prediction = None
    graph = None
    metrics = None

    if request.method == "POST":
        try:
            height = float(request.form["height"])
            weight = float(request.form["weight"])

            prediction = calculate_regression(height, weight)
            graph = generate_graph_regression()
            metrics = obtener_metricas()

        except Exception as e:
            return f"Error en regresión: {e}"

    return render_template(
        "regression.html",
        prediction=prediction,
        graph=graph,
        metrics=metrics
    )

@app.route("/logistic", methods=["GET", "POST"])
def logistic_view():
    prediction = None
    metrics = obtener_datos()  # <-- siempre cargamos columnas
    roc_graph = None

    if request.method == "POST":
        try:
            columnas = metrics["columnas"]
            datos = [float(request.form[col]) for col in columnas]

            prediction = predecir(datos)
            roc_graph = obtener_roc_graph()

        except Exception as e:
            return f"Error en logística: {e}"

    return render_template(
        "logistic.html",
        prediction=prediction,
        metrics=metrics,
        roc_graph=roc_graph
    )
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
