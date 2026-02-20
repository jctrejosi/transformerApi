from fastapi import FastAPI, BackgroundTasks, HTTPException
from .train import run_full_train
from .fine_tuning import run_fine_tuning
from .predict import WeatherPredictor
from utils.tools import dotdict
import os, pickle

app = FastAPI()

# Configuración (Asegúrate de que coincida con tus datos)
config = dotdict({'seq_len': 96, 'pred_len': 24, 'enc_in': 3, 'd_model': 512, 'n_heads': 8, 'e_layers': 2, 'd_ff': 2048, 'dropout': 0.1, 'output_attention': False})

MODEL_PATH = "weights/bitcoin_model.pth"
SCALER_PATH = "weights/scaler.pkl"
predictor = None

def reload_predictor():
    global predictor
    if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
        with open(SCALER_PATH, 'rb') as f:
            scaler = pickle.load(f)
        predictor = WeatherPredictor(MODEL_PATH, config, scaler)

@app.on_event("startup")
async def startup(): reload_predictor()

@app.post("/train")
async def full_train(payload: dict, bg: BackgroundTasks):
    # 1. Obtener el nombre del modelo del JSON (ej: {"model_name": "clima_medellin", "data": [...]})
    model_id = payload.get('model_name', 'default_model')
    
    # 2. Construir rutas personalizadas
    custom_model_path = f"weights/{model_id}.pth"
    custom_scaler_path = f"weights/scaler_{model_id}.pkl"

    def task():
        # Pasamos las rutas dinámicas a la función de entrenamiento
        run_full_train(payload['data'], config, custom_model_path, custom_scaler_path)
        
        # OJO: Aquí actualizamos las variables globales para que el predictor 
        # use el último modelo entrenado
        global MODEL_PATH, SCALER_PATH
        MODEL_PATH = custom_model_path
        SCALER_PATH = custom_scaler_path
        reload_predictor()

    bg.add_task(task)
    return {"status": f"Iniciando entrenamiento del modelo: {model_id}"}

@app.post("/fine_tuning")
async def fine_tune(payload: dict, bg: BackgroundTasks):
    """Actualiza un modelo específico con datos nuevos (Fine-Tuning)."""
    # 1. Obtener nombre del modelo y rutas
    model_id = payload.get('model_name', 'default_model')
    target_model = f"weights/{model_id}.pth"
    target_scaler = f"weights/scaler_{model_id}.pkl"

    # 2. VALIDACIÓN: No se puede ajustar algo que no existe
    if not os.path.exists(target_model):
        raise HTTPException(
            status_code=400,
            detail=f"El modelo '{model_id}' no existe. Debes entrenarlo primero con /train."
        )

    def task():
        # 3. Ejecutar el fine tuning sobre el archivo específico
        run_fine_tuning(payload['data'], config, target_model, target_scaler)

        # 4. Recargar el predictor global si el modelo ajustado es el que está en uso
        global MODEL_PATH, SCALER_PATH
        MODEL_PATH = target_model
        SCALER_PATH = target_scaler
        reload_predictor()

    bg.add_task(task)
    return {"status": f"Iniciando Ajuste Fino para el modelo: {model_id}"}

@app.post("/predict")
async def get_prediction(payload: dict):
    try:
        model_id = payload.get('model_name', 'default_model')
        target_model = f"weights/{model_id}.pth"
        target_scaler = f"weights/scaler_{model_id}.pkl"

        if not os.path.exists(target_model) or not os.path.exists(target_scaler):
            raise HTTPException(
                status_code=404, 
                detail=f"El modelo '{model_id}' no existe."
            )

        with open(target_scaler, 'rb') as f:
            scaler = pickle.load(f)
        
        current_predictor = WeatherPredictor(target_model, config, scaler)
        
        # --- CAMBIO AQUÍ: Ahora extraemos de 'data' en lugar de 'history' ---
        historical_data = payload.get('data')
        n_points = payload.get('points', config.pred_len)

        # Si me mandan más de 96, corto y me quedo con los últimos 96
        if len(historical_data) > config.seq_len:
            historical_data = historical_data[-config.seq_len:]
        if not historical_data or len(historical_data) < config.seq_len:
            return {
                "status": "error", 
                "message": f"Se requieren exactamente {config.seq_len} registros en 'data'."
            }

        # Pasamos esos datos al predictor
        forecast = current_predictor.predict(historical_data, points_to_return=n_points)
        
        return {
            "status": "success",
            "model_used": model_id,
            "forecast": forecast
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}