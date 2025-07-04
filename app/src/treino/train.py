import numpy as np
import os
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Caminhos
DATA_DIR = "app/data"
MODEL_OUTPUT_PATH = os.path.join(DATA_DIR, "logistic_model.joblib")

# 1. Carregar os dados vetorizados
train_data = np.load(os.path.join(DATA_DIR, "train_data.npz"), allow_pickle=True)
X_train, y_train = train_data["X"].item(), train_data["y"]

# 2. Treinar o modelo
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# 3. Avaliar o modelo (com dados de teste)
test_data = np.load(os.path.join(DATA_DIR, "test_data.npz"), allow_pickle=True)
X_test, y_test = test_data["X"].item(), test_data["y"]

y_pred = model.predict(X_test)

print("\n[Relatório de Classificação]")
print(classification_report(y_test, y_pred, target_names=["true", "fake"]))

print("\n[Matriz de Confusão]")
print(confusion_matrix(y_test, y_pred))

# 4. Salvar o modelo treinado
joblib.dump(model, MODEL_OUTPUT_PATH)
print(f"\n[OK] Modelo salvo em: {MODEL_OUTPUT_PATH}")
