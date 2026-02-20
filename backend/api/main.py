from fastapi import FastAPI, BackgroundTasks, HTTPException, UploadFile, File
from .fine_tuning import run_fine_tuning
from .predict import WeatherPredictor
from utils.tools import dotdict
import os, pickle

from .upload_model import router as upload_router
from .download_model import router as download_router
from .train import router as train_router
from .predict import router as predict_router
from .fine_tuning import router as fine_tuning_router

app = FastAPI()

app.include_router(upload_router)
app.include_router(download_router)
app.include_router(train_router)
app.include_router(predict_router)
app.include_router(fine_tuning_router)
