import zipfile
import io
from fastapi import APIRouter, HTTPException, UploadFile, File
from pathlib import Path

router = APIRouter(tags=["upload"])
MODEL_DIR = Path("saved_models")

@router.post("/upload_model")
async def upload_model_package(file: UploadFile = File(...)):
    if not file.filename.endswith('.zip'):
        raise HTTPException(status_code=400, detail="El archivo debe ser un .zip")

    try:
        contents = await file.read()
        z = zipfile.ZipFile(io.BytesIO(contents))
        
        filenames = z.namelist()
        has_pth = any(f.endswith('.pth') for f in filenames)
        has_pkl = any(f.endswith('.pkl') for f in filenames)
        
        if not (has_pth and has_pkl):
            raise HTTPException(status_code=400, detail="El ZIP debe contener .pth y .pkl")

        z.extractall(MODEL_DIR)
        
        return {
            "status": "success",
            "message": f"Modelo '{file.filename}' instalado correctamente",
            "files": filenames
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))