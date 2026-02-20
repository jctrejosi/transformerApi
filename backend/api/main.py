from fastapi import FastAPI, BackgroundTasks, HTTPException
from .train import run_full_train
from .fine_tuning import run_fine_tuning
from .predict import WeatherPredictor
from utils.tools import dotdict
import os, pickle
import shutil
from fastapi.responses import FileResponse
from pathlib import Path

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
    custom_scaler_path = f"weights/{model_id}.pkl"

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
    target_scaler = f"weights/{model_id}.pkl"

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
        target_scaler = f"weights/{model_id}.pkl"

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

@app.post("/download_model")
async def download_model_package(payload: dict, background_tasks: BackgroundTasks):
    """
    Recibe {"model_name": "nombre"} y devuelve un ZIP con el .pth y el .pkl
    """
    model_name = payload.get("model_name")
    if not model_name:
        raise HTTPException(status_code=400, detail="Falta 'model_name' en el payload")

    model_path = Path(f"weights/{model_name}.pth")
    scaler_path = Path(f"weights/{model_name}.pkl")
    
    # Carpeta temporal para el empaquetado
    export_path = Path(f"exports/{model_name}")
    zip_full_path = Path(f"exports/{model_name}_pack") # shutil añade el .zip automáticamente

    # 1. Validar existencia de la pareja
    if not model_path.exists() or not scaler_path.exists():
        raise HTTPException(
            status_code=404, 
            detail=f"No se encontró la pareja (.pth y .pkl) para el modelo: {model_name}"
        )

    try:
        # 2. Crear estructura temporal
        os.makedirs("exports", exist_ok=True)
        if export_path.exists(): shutil.rmtree(export_path)
        export_path.mkdir()

        # 3. Copiar archivos al área de exportación
        shutil.copy2(model_path, export_path / f"{model_name}.pth")
        shutil.copy2(scaler_path, export_path / f"{model_name}.pkl")

        # 4. Crear el ZIP
        # make_archive(nombre_archivo, formato, directorio_a_comprimir)
        shutil.make_archive(str(zip_full_path), 'zip', export_path)

        # 5. Limpiar carpeta temporal (ya no la necesitamos, tenemos el zip)
        shutil.rmtree(export_path)

        # 6. Tarea en segundo plano para borrar el ZIP después de enviarlo (limpieza)
        final_zip = Path(f"{zip_full_path}.zip")
        background_tasks.add_task(os.remove, final_zip)

        return FileResponse(
            path=final_zip,
            filename=f"{model_name}_complete.zip",
            media_type='application/zip'
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al generar paquete: {str(e)}")