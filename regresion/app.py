from flask import Flask, render_template, request
from logistic import predecir, obtener_datos, obtener_roc_graph
from regression import calculate_grade, generate_graph

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/regression", methods=["GET", "POST"])
def regression_view():
    prediction = None
    graph = None

    if request.method == "POST":
        hours = float(request.form["hours"])
        prediction = calculate_grade(hours)
        graph = generate_graph()

    return render_template("regression.html", prediction=prediction, graph=graph)

@app.route("/logistic", methods=["GET", "POST"])
def logistic_view():
    prediction = None
    metrics = None
    roc_graph = None

    if request.method == "POST":
        try:
            datos = [
                float(request.form["edad"]),
                float(request.form["ingreso_mensual"]),
                float(request.form["visitas_web_mes"]),
                float(request.form["tiempo_sitio_min"]),
                float(request.form["compras_previas"]),
                float(request.form["descuento_usado"])
            ]

            prediction = predecir(datos)
            metrics = obtener_datos()
            roc_graph = obtener_roc_graph()

        except Exception as e:
            return f"Error: {e}"

    return render_template("logistic.html", prediction=prediction, metrics=metrics, roc_graph=roc_graph)

if __name__ == "__main__":
    app.run(debug=True)
