import numpy as np
import os
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

# Caminhos
DATA_PATH = "app/data_bert/bert_embeddings.npz"
MODEL_OUTPUT = "app/data_bert/bert_logistic_model.joblib"

# 1. Carregar os dados
print("[INFO] Carregando embeddings...")
data = np.load(DATA_PATH)
X = data["X"]
y = data["y"]

# 2. Dividir treino e teste
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3. Treinar o modelo
print("[INFO] Treinando modelo Logistic Regression...")
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# 4. Avaliar o modelo
y_pred = model.predict(X_test)
print("\n[Relatório de Classificação]")
print(classification_report(y_test, y_pred, target_names=["true", "fake"]))
print("\n[Matriz de Confusão]")
print(confusion_matrix(y_test, y_pred))

# 5. Salvar o modelo treinado
joblib.dump(model, MODEL_OUTPUT)
print(f"\n[OK] Modelo salvo em: {MODEL_OUTPUT}")
