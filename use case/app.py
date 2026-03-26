from flask import Flask, render_template, request

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

if __name__ == "__main__":
    app.run(debug=True)
