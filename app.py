import joblib, pandas as pd, os
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
from typing import Dict, Any

API_KEY = os.getenv("PREDICT_KEY", "dev-key")
pipe = joblib.load("barley_model/model.pkl")
TARGETS = ["grain_protein_pct","yield_t_ha"]

class PredictIn(BaseModel):
    features: Dict[str, Any]

app = FastAPI()

@app.post("/predict")
def predict(inp: PredictIn, x_api_key: str = Header(None)):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")
    df = pd.DataFrame([inp.features])
    y = pipe.predict(df)[0].tolist()
    return {"targets": TARGETS, "pred": y}
