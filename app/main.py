from fastapi import FastAPI
from app.routers import recomendacoes

app = FastAPI(title="Fake News Detector API")

app.include_router(recomendacoes.router)

@app.get("/")
def read_root():
    return {"message": "API de Detecção de Fake News está no ar"}
