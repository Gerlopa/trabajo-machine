import pandas as pd
import matplotlib.pyplot as plt
import io
import base64
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, roc_curve, auc
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample

# -------------------- CARGAR DATASET --------------------
data = pd.read_csv('data/health_activity_data.csv')

# -------------------- PREPARACIÓN --------------------
data = data.drop(columns=["ID"])

# Separar presión arterial
data[['Systolic', 'Diastolic']] = data['Blood_Pressure'].str.split('/', expand=True)
data['Systolic'] = pd.to_numeric(data['Systolic'])
data['Diastolic'] = pd.to_numeric(data['Diastolic'])
data = data.drop(columns=['Blood_Pressure'])

# Convertir categóricas
data["Gender"] = data["Gender"].map({"Male":0, "Female":1})
data["Smoker"] = data["Smoker"].map({"No":0, "Yes":1})
data["Diabetic"] = data["Diabetic"].map({"No":0, "Yes":1})
data["Heart_Disease"] = data["Heart_Disease"].map({"No":0, "Yes":1})

data = data.dropna()

# -------------------- BALANCEO --------------------
df_majority = data[data.Heart_Disease == 0]
df_minority = data[data.Heart_Disease == 1]

df_minority_upsampled = resample(
    df_minority,
    replace=True,
    n_samples=len(df_majority),
    random_state=42
)

data = pd.concat([df_majority, df_minority_upsampled])

# -------------------- VARIABLES --------------------
X = data.drop('Heart_Disease', axis=1)
y = data['Heart_Disease']

# División
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Escalado
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# -------------------- MODELO --------------------
qda_model = QuadraticDiscriminantAnalysis(reg_param=0.2)
qda_model.fit(X_train, y_train)

# -------------------- PREDICCIÓN --------------------
y_prob = qda_model.predict_proba(X_test)[:, 1]

threshold = 0.4
y_pred = (y_prob > threshold).astype(int)

# -------------------- MÉTRICAS --------------------
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)
cm = confusion_matrix(y_test, y_pred)
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

# -------------------- FUNCIONES --------------------

def predecir_qda(input_dict):
    df = pd.DataFrame([input_dict])
    df = scaler.transform(df)
    prob = qda_model.predict_proba(df)[0][1]
    return int(prob > threshold)


def datos_qda():
    tn, fp, fn, tp = cm.ravel()

    return {
        "accuracy": round(accuracy, 2),
        "precision": round(precision, 2),
        "recall": round(recall, 2),
        "f1": round(f1, 2),
        "auc": round(roc_auc, 2),

        "cm": cm.tolist(),

        "cm_values": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp)
        },

        "columnas": X.columns.tolist()
    }


def roc_qda():
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
    ax.plot([0, 1], [0, 1], '--')
    ax.set_xlabel('FPR')
    ax.set_ylabel('TPR')
    ax.set_title('ROC Curve - QDA')
    ax.legend()

    img = io.BytesIO()
    fig.savefig(img, format='png')
    plt.close(fig)
    img.seek(0)

    return base64.b64encode(img.getvalue()).decode()


def grafica_decision_qda():
    # 🔥 Variables para visualización
    X_vis = data[["BMI", "Age"]]
    y_vis = data["Heart_Disease"]

    # Escalado para consistencia
    scaler_vis = StandardScaler()
    X_vis_scaled = scaler_vis.fit_transform(X_vis)

    # Modelo para visualización
    model_vis = QuadraticDiscriminantAnalysis(reg_param=0.2)
    model_vis.fit(X_vis_scaled, y_vis)

    # Malla
    x_min, x_max = X_vis_scaled[:, 0].min() - 1, X_vis_scaled[:, 0].max() + 1
    y_min, y_max = X_vis_scaled[:, 1].min() - 1, X_vis_scaled[:, 1].max() + 1

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 200),
        np.linspace(y_min, y_max, 200)
    )

    Z = model_vis.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Gráfica
    fig, ax = plt.subplots()

    ax.contourf(xx, yy, Z, alpha=0.3, cmap="coolwarm")

    ax.scatter(
        X_vis_scaled[:, 0],
        X_vis_scaled[:, 1],
        c=y_vis,
        s=20,
        edgecolor='k'
    )

    ax.set_xlabel("BMI (scaled)")
    ax.set_ylabel("Age (scaled)")
    ax.set_title("QDA Decision Boundary")

    img = io.BytesIO()
    fig.savefig(img, format='png')
    plt.close(fig)
    img.seek(0)

    return base64.b64encode(img.getvalue()).decode()