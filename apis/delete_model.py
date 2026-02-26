import os
from fastapi import APIRouter, HTTPException
from pathlib import Path

router = APIRouter(tags=["inventory"])
MODEL_DIR = Path("saved_models")

@router.delete("/delete_model/{model_name}")
async def delete_model(model_name: str):
    """
    Elimina físicamente los archivos .pth y .pkl de un modelo específico.
    """
    pth_file = MODEL_DIR / f"{model_name}.pth"
    pkl_file = MODEL_DIR / f"{model_name}.pkl"
    
    deleted_files = []
    
    # Intentar borrar el archivo .pth
    if pth_file.exists():
        os.remove(pth_file)
        deleted_files.append(pth_file.name)
        
    # Intentar borrar el archivo .pkl
    if pkl_file.exists():
        os.remove(pkl_file)
        deleted_files.append(pkl_file.name)

    if not deleted_files:
        raise HTTPException(
            status_code=404, 
            detail=f"No se encontraron archivos para el modelo '{model_name}'."
        )

    return {
        "status": "success",
        "message": f"Modelo '{model_name}' eliminado correctamente.",
        "deleted_files": deleted_files
    }