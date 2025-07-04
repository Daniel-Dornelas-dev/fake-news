import joblib
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel

# Caminhos
MODEL_PATH = "app/data_bert/bert_logistic_model.joblib"
MODEL_NAME = "neuralmind/bert-base-portuguese-cased"

# Carregar modelo e tokenizer
print("[INFO] Carregando modelo e tokenizer BERTimbau...")
model = joblib.load(MODEL_PATH)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
bert = AutoModel.from_pretrained(MODEL_NAME)
bert.eval()

def gerar_embedding(texto):
    inputs = tokenizer(texto, return_tensors="pt", truncation=True, padding="max_length", max_length=128)
    with torch.no_grad():
        outputs = bert(**inputs)
        cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze().numpy()
    return cls_embedding.reshape(1, -1)

def classificar_frase(frase):
    embedding = gerar_embedding(frase)
    pred = model.predict(embedding)[0]
    prob = model.predict_proba(embedding)[0][pred]
    label = "Fake" if pred == 1 else "True"
    return label, round(prob * 100, 2)

if __name__ == "__main__":
    print("\n[Pronto] Classificador BERTimbau carregado.\n")
    while True:
        entrada = input("Digite uma frase para classificar ('sair' para encerrar): ")
        if entrada.strip().lower() in ["sair", "exit"]:
            break
        resultado, confianca = classificar_frase(entrada)
        print(f"\nClassificação: {resultado} ({confianca}% de confiança)\n")