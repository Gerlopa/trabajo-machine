
import li_regression

from flask import Flask, render_template, request

app = Flask(__name__)

@app.route("/")
def inicio():
    return render_template("index.html", page="home")

@app.route("/proyecto")
def proyecto():
    use_cases = [
        {"title": "Fraud Detection",
         "desc": "Machine learning models analyze transaction patterns "
                 "such as amount, location, and frequency to detect "
                 "suspicious or fraudulent banking activities."},

        {"title": "Recommendation Systems",
         "desc": "Algorithms analyze user behavior and preferences to "
                 "recommend products, movies, or content, improving "
                 "personalization and user engagement."},

        {"title": "Facial Recognition",
         "desc": "Computer vision models analyze facial features in "
                 "images or videos to identify or verify a person."},

        {"title": "Medical Diagnosis",
         "desc": "Machine learning analyzes medical data and images "
                 "to help doctors detect diseases and support "
                 "clinical decision-making."},

        {"title": "Natural Language Processing",
         "desc": "NLP models allow computers to understand and "
                 "generate human language for applications such as "
                 "chatbots, translators, and virtual assistants."},

        {"title": "Demand Prediction",
         "desc": "Regression and time series models analyze historical "
                 "data to forecast future product demand and support "
                 "business planning."},

        {"title": "Autonomous Driving",
         "desc": "Self-driving systems use sensors, cameras, and "
                 "deep learning models to interpret the environment "
                 "and make driving decisions in real time."},

        {"title": "Email Classification",
         "desc": "Supervised learning models classify emails into "
                 "categories such as spam, promotions, or important "
                 "messages."},

        {"title": "Sentiment Analysis",
         "desc": "NLP techniques analyze text data to determine "
                 "whether opinions are positive, negative, or neutral."},

        {"title": "Predictive Maintenance",
         "desc": "Machine learning analyzes equipment data to predict "
                 "failures before they happen, reducing downtime and "
                 "maintenance costs."}
    ]

    return render_template("index.html", page="project", use_cases=use_cases)

@app.route("/FirstPage/<name>")
def first_page(name=""):
    return render_template("index.html", page="home", name=name)

@app.route("/SecondPage/<name>")
def second_page(name=""):
    return render_template("index.html", page="project", name=name,
                           use_cases=[])
@app.route("/li_regression", methods=["GET","POST"])
def calculate_grade_route():

    result = None

    if request.method == "POST":
        hours = float(request.form["hours"])
        result = li_regression.calculate_grade(hours)

    return render_template("li_regresion.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)

