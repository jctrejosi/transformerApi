import torch
import pickle
import os
from fastapi import APIRouter, BackgroundTasks, HTTPException
from torch.utils.data import DataLoader
from models.iTransformer import Model
from utils.convertJson import JSONDataset

router = APIRouter(tags=["fine_tuning"])

# --- FUNCIÓN DE LÓGICA DE AJUSTE FINO ---
def execute_fine_tuning_logic(data_json, config, model_path, scaler_path):
    try:
        # 1. Cargar Scaler existente (NO SE RE-ENTRENA)
        if not os.path.exists(scaler_path):
            print(f"Error: Scaler no encontrado en {scaler_path}")
            return

        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        
        # 2. Preparar datos con el scaler viejo
        dataset = JSONDataset(data_json, config, scaler=scaler)
        loader = DataLoader(dataset, batch_size=8, shuffle=True)
        
        # 3. Cargar Modelo existente
        if not os.path.exists(model_path):
            print(f"Error: Modelo no encontrado en {model_path}")
            return

        model = Model(config)
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
        
        # 4. Configuración de ajuste (LR muy bajo)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-5) 
        criterion = torch.nn.MSELoss()

        model.train()
        for epoch in range(2): # 2 pasadas rápidas
            for batch_x, batch_y, batch_x_m, batch_y_m in loader:
                optimizer.zero_grad()
                outputs = model(batch_x, batch_x_m, torch.zeros_like(batch_y), batch_y_m)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
        
        # 5. Guardar el modelo actualizado
        torch.save(model.state_dict(), model_path)
        print(f"-> Fine-Tuning completado para: {model_path}")

    except Exception as e:
        print(f"Error en Fine-Tuning: {str(e)}")

# --- ENDPOINT DE LA API ---
@router.post("/fine_tuning")
async def fine_tune_endpoint(payload: dict, bg: BackgroundTasks):
    from api.main import config  # Importar config global

    model_id = payload.get('model_name', 'default_model')
    data_json = payload.get('data')

    if not data_json:
        raise HTTPException(status_code=400, detail="Faltan datos en 'data'")

    target_model = f"saved_models/{model_id}.pth"
    target_scaler = f"saved_models/{model_id}.pkl"

    # Validación previa: No podemos tunear lo que no existe
    if not os.path.exists(target_model):
        raise HTTPException(
            status_code=404, 
            detail=f"El modelo '{model_id}' no existe. Debe entrenarse primero con /train."
        )

    # Ejecutar en segundo plano
    bg.add_task(execute_fine_tuning_logic, data_json, config, target_model, target_scaler)

    return {
        "status": "fine_tuning_started",
        "message": f"Ajuste fino en marcha para el modelo '{model_id}'."
    }