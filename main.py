from fastapi import FastAPI
from model import predecir

app = FastAPI()

@app.get("/")
def home():
    return {"status": "ML API activa"}

@app.post("/predict")
def predict(data: dict):
    
    prob = predecir(
        data["progreso"],
        data["dias_inactivo"],
        data["promedio_examen"]
    )
    
    return {"probabilidad_abandono": prob}