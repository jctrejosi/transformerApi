from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import datetime
import torch
import uvicorn  # <--- Asegúrate de tener esta importación

app = FastAPI(title="iTransformer Prediction Engine")

class PredictionRequest(BaseModel):
    model_id: str
    history: List[Dict[str, Any]]
    real_data_last_step: Optional[Dict[str, float]] = None

@app.get("/health")
def health_check():
    return {
        "status": "online",
        "timestamp": datetime.datetime.now().isoformat(),
        "torch_version": torch.__version__,
        "device": "cuda" if torch.cuda.is_available() else "cpu"
    }

@app.post("/predict")
async def predict(payload: PredictionRequest):
    # Por ahora es un placeholder
    return {"message": "Listo para recibir datos", "model": payload.model_id}

# --- ESTO ES LO QUE PERMITE EJECUTARLO CON 'PYTHON MAIN.PY' ---
if __name__ == "__main__":
    print("Iniciando servidor de predicción...")
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)