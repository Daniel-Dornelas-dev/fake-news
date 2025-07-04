import joblib
import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np

# Caminhos dos modelos
MODEL_PATH = "app/data_bert/bert_logistic_model.joblib"
MODEL_NAME = "neuralmind/bert-base-portuguese-cased"

# Carregamento único
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
bert = AutoModel.from_pretrained(MODEL_NAME)
bert.eval()
model = joblib.load(MODEL_PATH)

def gerar_embedding(texto: str):
    inputs = tokenizer(texto, return_tensors="pt", truncation=True, padding="max_length", max_length=128)
    with torch.no_grad():
        outputs = bert(**inputs)
        cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze().numpy()
    return cls_embedding.reshape(1, -1)

def classificar_texto(texto: str):
    embedding = gerar_embedding(texto)
    pred = model.predict(embedding)[0]
    prob = model.predict_proba(embedding)[0][pred]
    resultado = "Fake" if pred == 1 else "True"
    return resultado, round(prob * 100, 2)
