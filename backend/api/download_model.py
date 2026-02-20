import os
import shutil
from fastapi import APIRouter, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
from pathlib import Path

router = APIRouter(tags=["download"])
MODEL_DIR = Path("saved_models")

@router.post("/download_model")
async def download_model_package(payload: dict, background_tasks: BackgroundTasks):
    model_name = payload.get("model_name")
    if not model_name:
        raise HTTPException(status_code=400, detail="Falta 'model_name'")

    model_path = MODEL_DIR / f"{model_name}.pth"
    scaler_path = MODEL_DIR / f"{model_name}.pkl"
    
    if not model_path.exists() or not scaler_path.exists():
        raise HTTPException(status_code=404, detail="Pareja de archivos no encontrada")

    export_path = Path(f"exports/{model_name}")
    zip_full_path = Path(f"exports/{model_name}_pack")

    try:
        os.makedirs("exports", exist_ok=True)
        if export_path.exists(): shutil.rmtree(export_path)
        export_path.mkdir()

        shutil.copy2(model_path, export_path / f"{model_name}.pth")
        shutil.copy2(scaler_path, export_path / f"{model_name}.pkl")

        shutil.make_archive(str(zip_full_path), 'zip', export_path)
        shutil.rmtree(export_path)

        final_zip = Path(f"{zip_full_path}.zip")
        background_tasks.add_task(os.remove, final_zip)

        return FileResponse(
            path=final_zip,
            filename=f"{model_name}_complete.zip",
            media_type='application/zip'
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))