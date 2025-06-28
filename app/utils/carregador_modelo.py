import joblib
from pathlib import Path

CAMINHO_MODELO = Path("app/models/modelo_fake_news.joblib")

def carregar_modelo():
    if not CAMINHO_MODELO.exists():
        raise FileNotFoundError(f"Modelo não encontrado em: {CAMINHO_MODELO.resolve()}")
    return joblib.load(CAMINHO_MODELO)
