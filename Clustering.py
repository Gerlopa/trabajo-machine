
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

def getData():
       return [{"nombre": "Ana", "edad": 22, "ingresos": 1200, "gasto": 300},
        {"nombre": "Luis", "edad": 25, "ingresos": 1500, "gasto": 350},
        {"nombre": "Carlos", "edad": 23, "ingresos": 1300, "gasto": 280},
        {"nombre": "Marta", "edad": 45, "ingresos": 4000, "gasto": 1200},
        {"nombre": "Sofía", "edad": 50, "ingresos": 4200, "gasto": 1400},
        {"nombre": "Jorge", "edad": 47, "ingresos": 3900, "gasto": 1100},
        {"nombre": "Elena", "edad": 31, "ingresos": 2500, "gasto": 700},
        {"nombre": "Pedro", "edad": 33, "ingresos": 2700, "gasto": 750},
        {"nombre": "Laura", "edad": 29, "ingresos": 2400, "gasto": 680},
        {"nombre": "Andrés", "edad": 52, "ingresos": 5000, "gasto": 1600},
        {"nombre": "Camila", "edad": 21, "ingresos": 1100, "gasto": 250},
        {"nombre": "Diego", "edad": 38, "ingresos": 3200, "gasto": 900}]

def applyClustering():
    data = getData()
    
    X = [[person["edad"],person["ingresos"],person["gasto"]]for person in data]
    scaler = StandardScaler()
    Xscaled = scaler.fit_transform(X)
    model = KMeans (n_clusters=3, random_state=42, n_init=10)
    labels = model.fit_predict(Xscaled)
    
    
    result = []
    for i, person in enumerate(data):
        row = person.copy()
        row["Cluster"] = int(labels[i])
        result.append(row)
        
        clustersSumary = {}
        
        for label in labels:
            label = int(label)
            clustersSumary[label] = clustersSumary.get(label,0) + 1
            
        centers = model.cluster_centers_.tolist()
        
        return{
            "result" : result,
            "clustersSumary" : clustersSumary,
            "centers" : centers
        }
