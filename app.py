from fastapi import FastAPI
from prediction_finale import predict_next_leave

app = FastAPI()

@app.get("/predict")
def predict(Nom_Employe: str):
    return {"result": predict_next_leave(Nom_Employe)}
