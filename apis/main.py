from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .upload_model import router as upload_router
from .download_model import router as download_router
from .train import router as train_router
from .predict import router as predict_router
from .fine_tuning import router as fine_tuning_router
from .delete_all import router as delete_all
from .delete_model import router as delete_model
from .list_models import router as list_models

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "https://tu-frontend.vercel.app",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(upload_router)
app.include_router(download_router)
app.include_router(train_router)
app.include_router(predict_router)
app.include_router(fine_tuning_router)
app.include_router(delete_all)
app.include_router(delete_model)
app.include_router(list_models)


@app.get("/health")
def health():
    return {"status": "ok"}