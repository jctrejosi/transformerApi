import os
from fastapi import APIRouter
from pathlib import Path
from datetime import datetime

router = APIRouter(tags=["inventory"])
MODEL_DIR = Path("saved_models")

@router.get("/list_models")
async def list_models():
    """
    Escanea la carpeta de modelos y devuelve una lista de los modelos 
    completos (que tienen tanto .pth como .pkl).
    """
    if not MODEL_DIR.exists():
        return {"models": [], "count": 0}

    # Obtenemos todos los archivos en la carpeta
    files = os.listdir(MODEL_DIR)
    
    # Buscamos nombres base que tengan ambos archivos
    model_names = set(f.rsplit('.', 1)[0] for f in files if f.endswith(('.pth', '.pkl')))
    
    inventory = []
    for name in model_names:
        pth_file = MODEL_DIR / f"{name}.pth"
        pkl_file = MODEL_DIR / f"{name}.pkl"
        
        # Solo lo listamos si la pareja existe
        if pth_file.exists() and pkl_file.exists():
            stats = pth_file.stat()
            inventory.append({
                "model_name": name,
                "last_modified": datetime.fromtimestamp(stats.st_mtime).strftime('%Y-%m-%d %H:%M:%S'),
                "size_kb": round(stats.st_size / 1024, 2)
            })

    return {
        "models": inventory,
        "count": len(inventory)
    }