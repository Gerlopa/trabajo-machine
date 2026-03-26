import pandas as pd
import matplotlib.pyplot as plt
import io
import base64
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_curve, auc

#Cargar datos
data = pd.read_excel('data/dataset_regresion_logistica.xlsx')

# Corregir decimales
data = data.replace(',', '.', regex=True)
data = data.apply(pd.to_numeric)

# Variables
X = data.drop('target', axis=1)
y = data['target']

# División
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Modelo
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

#Evaluación
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

#ROC datos
y_prob = model.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

#Función para predecir
def predecir(valores):
    pred = model.predict([valores])
    return int(pred[0])

# Función para datos de gráfica
def obtener_datos():
    return {
        "accuracy": round(accuracy, 2),
        "fpr": fpr.tolist(),
        "tpr": tpr.tolist(),
        "auc": round(roc_auc, 2),
        "columnas": X.columns.tolist()
    }


def obtener_roc_graph():
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'AUC = {roc_auc:.2f}')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve - Logistic Regression')
    plt.legend(loc='lower right')

    img = io.BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight')
    plt.close()
    img.seek(0)
    return base64.b64encode(img.getvalue()).decode()