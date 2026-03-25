from flask import Flask, render_template, request
import LinearRegression
import LogisticRegressionModel


app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

# @app.route ('/')
# def home():
#   return ('Welcome to Machine Learning') 

@app.route('/FirstPage/<name>')
def FirstPage(name=''):
    return render_template('FirstPage.html', name=name)

@app.route("/LinearRegression/", methods = ["GET", "POST"])
def calculateGrade():
    calculateGrade = None 
    if request.method == "POST":
        hours = float(request.form["hours"])
        calculateGrade = LinearRegression.calculateGrade(hours)
    return render_template("linearRegressionGrade.html", result = calculateGrade)

@app.route('/logisticRegression', methods=["GET", "POST"])
def logisticRegression():
    result = None
    
    if request.method == "POST":
        edad = float(request.form["edad"])
        ingreso = float(request.form["ingreso"])
        visitas = float(request.form["visitas"])
        tiempo = float(request.form["tiempo"])
        compras = float(request.form["compras"])
        descuento = float(request.form["descuento"])

        data = [edad, ingreso, visitas, tiempo, compras, descuento]
        result = LogisticRegressionModel.predict(data)

    return render_template("logisticRegression.html", result=result)