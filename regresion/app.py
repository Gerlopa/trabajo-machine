from flask import Flask, render_template, request
from logistic import predecir, obtener_datos, obtener_roc_graph
from regression import calculate_grade, generate_graph

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/linear_explanation")
def linear_explanation():
    return render_template("linear_explanation.html")


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

if __name__ == "__main__":
    app.run(debug=True)
