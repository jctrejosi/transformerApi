import torch
import pickle
import os
from fastapi import APIRouter, BackgroundTasks, HTTPException
from torch.utils.data import DataLoader
from models.iTransformer import Model
from utils.convertJson import JSONDataset

# Creamos el router
router = APIRouter(tags=["training"])

# --- FUNCIÓN DE LÓGICA DE ENTRENAMIENTO ---
# Esta función es la que hace el trabajo pesado
def execute_training_logic(data_json, config, model_path, scaler_path):
    try:
        # 1. Preparar datos
        dataset = JSONDataset(data_json, config)
        loader = DataLoader(dataset, batch_size=32, shuffle=True)

        # 2. Configurar modelo (CPU por defecto)
        model = Model(config).to("cpu")
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = torch.nn.MSELoss()

        # 3. Bucle de entrenamiento
        model.train()
        for epoch in range(10):
            for batch_x, batch_y, batch_x_m, batch_y_m in loader:
                optimizer.zero_grad()
                # iTransformer suele requerir estos inputs
                outputs = model(batch_x, batch_x_m, torch.zeros_like(batch_y), batch_y_m)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

        # 4. Asegurar que la carpeta existe y guardar
        os.makedirs(os.path.dirname(model_path), exist_ok=True)

        # Guardar pesos (.pth)
        torch.save(model.state_dict(), model_path)

        # Guardar scaler (.pkl)
        with open(scaler_path, 'wb') as f:
            pickle.dump(dataset.scaler, f)

        print(f"-> Entrenamiento completado exitosamente: {model_path}")

    except Exception as e:
        print(f"Error crítico en el entrenamiento: {str(e)}")

# --- ENDPOINT DE LA API ---
@router.post("/train")
async def train_endpoint(payload: dict, bg: BackgroundTasks):
    """
    Recibe el JSON con 'data' y 'model_name'.
    Ejecuta el entrenamiento en segundo plano.
    """
    from api.main import config  # Importamos la config global de tu main

    data_json = payload.get('data')
    model_name = payload.get('model_name', 'default_model')

    if not data_json:
        raise HTTPException(status_code=400, detail="Faltan los datos ('data') en el JSON")

    # Definir rutas de archivos
    model_path = f"saved_models/{model_name}.pth"
    scaler_path = f"saved_models/{model_name}.pkl"

    # Lanzar la tarea en segundo plano para no bloquear al usuario
    bg.add_task(execute_training_logic, data_json, config, model_path, scaler_path)

    return {
        "status": "training_started",
        "message": f"El modelo '{model_name}' se está entrenando en segundo plano.",
        "assets": [model_path, scaler_path]
    }