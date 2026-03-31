import pandas as pd
import matplotlib.pyplot as plt
import io
import base64
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_curve, auc, confusion_matrix, precision_score, recall_score, f1_score
from imblearn.over_sampling import SMOTE  # Para balancear clases

# Cargar dataset
data = pd.read_csv('data/health_activity_data.csv')

#Eliminar ID
data = data.drop(columns=["ID"])

#Separar presión arterial
data[['Systolic', 'Diastolic']] = data['Blood_Pressure'].str.split('/', expand=True)
data['Systolic'] = pd.to_numeric(data['Systolic'])
data['Diastolic'] = pd.to_numeric(data['Diastolic'])
data = data.drop(columns=['Blood_Pressure'])

#Convertir variables categóricas
data["Gender"] = data["Gender"].map({"Male": 0, "Female": 1})
data["Smoker"] = data["Smoker"].map({"No": 0, "Yes": 1})
data["Diabetic"] = data["Diabetic"].map({"No": 0, "Yes": 1})
data["Heart_Disease"] = data["Heart_Disease"].map({"No": 0, "Yes": 1})

# Eliminar posibles nulos
data = data.dropna()

# Variables
X = data.drop('Heart_Disease', axis=1)
y = data['Heart_Disease']

# División
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

#SMOTE para balancear clases en el entrenamiento
sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

# Modelo
model = LogisticRegression(max_iter=1000)
model.fit(X_train_res, y_train_res)

# Predicción y métricas
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)
cm = confusion_matrix(y_test, y_pred)
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

#Función de predicción para frontend
def predecir(valores):
    pred = model.predict([valores])
    return int(pred[0])

#| Datos para el template
def obtener_datos():
    return {
        "accuracy": round(accuracy, 2),
        "precision": round(precision, 2),
        "recall": round(recall, 2),
        "f1": round(f1, 2),
        "fpr": fpr.tolist(),
        "tpr": tpr.tolist(),
        "auc": round(roc_auc, 2),
        "cm": cm.tolist(),
        "columnas": X.columns.tolist()
    }

#|Gráfica ROC
def obtener_roc_graph():
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, lw=2, label=f'AUC = {roc_auc:.2f}')
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve - Logistic Regression')
    plt.legend(loc='lower right')

    img = io.BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight')
    plt.close()
    img.seek(0)
    return base64.b64encode(img.getvalue()).decode()
