import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib

np.random.seed(42)
n = 500

data = pd.DataFrame({
    "progreso": np.random.rand(n),
    "dias_inactivo": np.random.randint(0, 30, n),
    "promedio_examen": np.random.randint(0, 20, n)
})

data["abandono"] = (
    (data["progreso"] < 0.4) &
    (data["dias_inactivo"] > 7)
).astype(int)

X = data.drop("abandono", axis=1)
y = data["abandono"]

model = RandomForestClassifier()
model.fit(X, y)

joblib.dump(model, "modelo.pkl")

print("modelo entrenado")