import pandas as pd
import matplotlib.pyplot as plt
import io
import base64
from sklearn.linear_model import LinearRegression
import matplotlib
matplotlib.use('Agg') 


df = pd.read_csv("data/health_activity_data.csv")

X = df[[
    "Age",
    "Height_cm"
]]

y = df["Weight_kg"]


model = LinearRegression()
model.fit(X, y)

def calculate_regression(age, height):
    data = [[age, height]]
    return model.predict(data)[0]

def generate_graph_regression():
    fig, ax = plt.subplots()

    ax.scatter(df["Height_cm"], y)

    # Promedio de edad
    avg_age = df["Age"].mean()

    X_line = pd.DataFrame({
        "Age": [avg_age] * len(df),
        "Height_cm": df["Height_cm"]
    })

    y_line = model.predict(X_line)

    ax.plot(df["Height_cm"], y_line)

    ax.set_xlabel("Height (cm)")
    ax.set_ylabel("Weight (kg)")

    img = io.BytesIO()
    fig.savefig(img, format='png')
    plt.close(fig)

    img.seek(0)
    return base64.b64encode(img.getvalue()).decode()
