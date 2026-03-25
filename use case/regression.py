import pandas as pd
import matplotlib.pyplot as plt
import io
import base64
from sklearn.linear_model import LinearRegression

# DATASET
data = {
    "Study Hours": [10, 15, 12, 8, 14, 5, 16, 7, 11, 13, 9, 4, 18, 3, 17, 6, 14, 2, 20, 1],
    "Final Grade": [3.8, 4.2, 3.6, 3, 4.5, 2.5, 4.8, 2.8, 3.7, 4, 3.2, 2.2, 5, 1.8, 4.9, 2.7, 4.4, 1.5, 5, 1]
}

df = pd.DataFrame(data)

X = df[["Study Hours"]]
y = df["Final Grade"]

model = LinearRegression()
model.fit(X, y)

def calculate_grade(study_hours):
    return model.predict([[study_hours]])[0]

def generate_graph():
    plt.figure()
    plt.scatter(X, y)
    plt.plot(X, model.predict(X))
    plt.xlabel("Study Hours")
    plt.ylabel("Final Grade")

    img = io.BytesIO()
    plt.savefig(img, format='png')
    plt.close()
    img.seek(0)

    return base64.b64encode(img.getvalue()).decode()