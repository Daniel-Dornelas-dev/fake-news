from transformers import AutoTokenizer, AutoModel
import torch
from tqdm import tqdm
import numpy as np

tokenizer = AutoTokenizer.from_pretrained("neuralmind/bert-base-portuguese-cased")
model = AutoModel.from_pretrained("neuralmind/bert-base-portuguese-cased")
model.eval()  # Modo avaliação

def gerar_embedding(texto):
    tokens = tokenizer(texto, return_tensors="pt", truncation=True, max_length=128, padding="max_length")
    with torch.no_grad():
        saida = model(**tokens)
    cls_embedding = saida.last_hidden_state[:, 0, :].squeeze().numpy()  # usamos o [CLS]
    return cls_embedding
