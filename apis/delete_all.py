import os
import shutil
from fastapi import APIRouter, HTTPException, Header
from pathlib import Path

router = APIRouter(tags=["inventory"])
MODEL_DIR = Path("saved_models")

@router.delete("/clear_all_models")
async def clear_all_models(x_admin_key: str = Header(None)):
    """
    Elimina TODOS los archivos de la carpeta de modelos.
    Requiere una clave de administración definida en variables de entorno.
    """
    # Obtenemos la clave desde la variable de entorno
    MASTER_KEY = os.getenv("CLEANUP_API_KEY", "super-secret-default-key")

    if x_admin_key != MASTER_KEY:
        raise HTTPException(
            status_code=401, 
            detail="No autorizado. Clave de administración incorrecta o ausente."
        )

    if not MODEL_DIR.exists():
        return {"status": "success", "message": "La carpeta ya está vacía."}

    try:
        files_to_delete = os.listdir(MODEL_DIR)
        for filename in files_to_delete:
            file_path = MODEL_DIR / filename
            if file_path.is_file(): os.remove(file_path)
            elif file_path.is_dir(): shutil.rmtree(file_path)

        return {
            "status": "success",
            "message": f"Limpieza completa realizada con éxito."
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))