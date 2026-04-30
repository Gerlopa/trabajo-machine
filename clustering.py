import pandas as pd
import matplotlib
matplotlib.use('Agg')

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import io
import base64

def clustering_completo():

    df = pd.read_csv('data/health_activity_data.csv')

    X = df[['Age', 'Weight_kg']]

    scaler = StandardScaler()
    Xscaled = scaler.fit_transform(X)

    iteraciones = []

    for i in [1, 2, 3]:

        model = KMeans(n_clusters=3, max_iter=i, random_state=42, n_init=1)
        labels = model.fit_predict(Xscaled)

        # Centroides en escala real
        centroids_scaled = model.cluster_centers_
        centroids = scaler.inverse_transform(centroids_scaled)

        inertia = model.inertia_

        # 📊 gráfica
        plt.figure()
        plt.scatter(df['Age'], df['Weight_kg'], c=labels, cmap='viridis', alpha=0.6)
        plt.scatter(centroids[:, 0], centroids[:, 1], marker='X', s=200)

        plt.title(f"Iteración {i}")
        plt.xlabel("Age")
        plt.ylabel("Weight")

        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close()
        buf.seek(0)

        graph = base64.b64encode(buf.getvalue()).decode()

        iteraciones.append({
            "iteracion": i,
            "centroides": centroids.tolist(),
            "inercia": inertia,
            "graph": graph
        })

    # Resultado final
    final_model = KMeans(n_clusters=3, random_state=42, n_init=10)
    df['Cluster'] = final_model.fit_predict(Xscaled)

    result = df.to_dict(orient='records')

    return {
        "iteraciones": iteraciones,
        "result": result
    }
