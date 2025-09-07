from fastapi import FastAPI, Query
import os
import pandas as pd
from prediction_finale import predict_next_leave  # 👈 import your function

# Create the FastAPI app
app = FastAPI()

@app.get("/predict")
def predict(name: str = Query(..., description="Nom complet de l'employé")):
    try:
        result = predict_next_leave(name)
        return {"employee": name, "result": result}
    except Exception as e:
        return {"error": str(e)}
