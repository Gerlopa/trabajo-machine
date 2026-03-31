import pandas as pd
import matplotlib.pyplot as plt
import io
import base64
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import matplotlib
matplotlib.use('Agg')

df = pd.read_csv("data/health_activity_data.csv")

df.columns = df.columns.str.strip()
df = df.replace(',', '.', regex=True)
df = df.apply(pd.to_numeric, errors='coerce')

df = df.rename(columns={
    "Height_cm": "Height",
    "Weight_kg": "Weight",
    "BMI": "BMI"
})

# Selección
df = df[["Height", "Weight", "BMI"]].dropna()

# Variables
X = df[["Height", "Weight"]]
y = df["BMI"]

# Modelo
model = LinearRegression()
model.fit(X, y)

# Predicción
def calculate_regression(height, weight):
    return round(float(model.predict([[height, weight]])[0]), 2)

# Métricas
def obtener_metricas():
    y_pred = model.predict(X)
    return {"r2": round(r2_score(y, y_pred), 2)}

# Gráfica
def generate_graph_regression():
    fig, ax = plt.subplots()

    df_sample = df.sample(200)

    ax.scatter(
        df_sample["Weight"],
        df_sample["BMI"],
        s=10,
        alpha=0.4
    )

    x_smooth = np.linspace(df["Weight"].min(), df["Weight"].max(), 100)

    avg_height = df["Height"].mean()

    X_line = pd.DataFrame({
        "Height": [avg_height] * len(x_smooth),
        "Weight": x_smooth
    })

    y_smooth = model.predict(X_line)

    ax.plot(x_smooth, y_smooth, linewidth=3)

    # Mostrar R²
    r2 = r2_score(y, model.predict(X))
    ax.text(0.05, 0.95, f"R² = {r2:.2f}",
            transform=ax.transAxes,
            verticalalignment='top')

    ax.set_xlabel("Weight (kg)")
    ax.set_ylabel("BMI")

    img = io.BytesIO()
    fig.savefig(img, format='png')
    plt.close(fig)

    img.seek(0)
    return base64.b64encode(img.getvalue()).decode()
