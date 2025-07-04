import joblib
import os
import numpy as np

# Caminhos
DATA_DIR = "app/data"
VECTORIZER_PATH = os.path.join(DATA_DIR, "tfidf_vectorizer.joblib")
MODEL_PATH = os.path.join(DATA_DIR, "logistic_model.joblib")

# Carregar modelo e vetorizer
vectorizer = joblib.load(VECTORIZER_PATH)
model = joblib.load(MODEL_PATH)

def classificar_noticia(texto: str):
    texto_vetorizado = vectorizer.transform([texto])
    predicao = model.predict(texto_vetorizado)[0]
    prob = model.predict_proba(texto_vetorizado)[0][predicao]
    return "Fake" if predicao == 1 else "True", round(prob * 100, 2)

if __name__ == "__main__":
    while True:
        entrada = input("Digite uma frase para classificar ('sair' para encerrar): ")
        if entrada.lower() in ["sair", "exit"]:
            break
        classe, probabilidade = classificar_noticia(entrada)
        print(f"\nClassificação: {classe} ({probabilidade}% de confiança)\n")
