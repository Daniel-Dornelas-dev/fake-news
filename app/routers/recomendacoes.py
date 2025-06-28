from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.services.recomendacao_service import classificar


router = APIRouter(prefix="/api", tags=["Recomendações"])

class NoticiaInput(BaseModel):
    texto: str

class ClassificacaoOutput(BaseModel):
    texto: str
    fake: bool
    probabilidade: float

@router.post("/classificar-noticia", response_model=ClassificacaoOutput)
def classificar_noticia(noticia: NoticiaInput):
    if not noticia.texto.strip():
        raise HTTPException(status_code=400, detail="Texto da notícia não pode ser vazio")

    try:
        resultado = classificar(noticia.texto)
        return resultado
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro na classificação: {str(e)}")
