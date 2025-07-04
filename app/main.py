from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from datetime import datetime

from app.src.services.classificador import classificar_texto
from app.src.services.historico import Historico
from app.src.services.status import obter_status_modelo


app = FastAPI(title="API de Detecção de Fake News", version="1.0")

# Modelo de entrada
class TextoEntrada(BaseModel):
    texto: str

@app.post("/api/classificar-noticia")
def classificar_noticia(entrada: TextoEntrada):
    print(f"📩 Texto recebido: {entrada.texto!r}")

    if not entrada.texto.strip():
        print("❌ Texto vazio recebido. Requisição rejeitada.")
        raise HTTPException(status_code=400, detail="Texto não pode ser vazio.")

    resultado, confianca = classificar_texto(entrada.texto)

    print(f"✅ Classificação: {resultado}, Confiança: {confianca:.2f}%")

    Historico.adicionar(entrada.texto, resultado, confianca)

    resposta = {
        "classificacao": resultado,
        "confianca": confianca,
        "data": datetime.now().isoformat()
    }

    print(f"📤 Resposta enviada: {resposta}")
    return resposta

# Rota para consultar o histórico
@app.get("/api/historico")
def consultar_historico():
    return Historico.listar()

# Rota para status do modelo
@app.get("/api/status")
def status_modelo():
    return obter_status_modelo()