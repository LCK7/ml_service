import joblib
import numpy as np

modelo = joblib.load("modelo.pkl")

def predecir(progreso, dias_inactivo, promedio_examen):
    datos = np.array([[progreso, dias_inactivo, promedio_examen]])
    prob = modelo.predict_proba(datos)[0][1]
    return float(prob)