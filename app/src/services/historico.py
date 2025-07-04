
from datetime import datetime

class Historico:
    _historico = []

    @classmethod
    def adicionar(cls, texto, resultado, confianca):
        cls._historico.append({
            "texto": texto,
            "classificacao": resultado,
            "confianca": confianca,
            "data": datetime.now().isoformat()
        })

    @classmethod
    def listar(cls):
        return list(reversed(cls._historico))  # mais recente primeiro
