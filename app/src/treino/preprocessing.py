import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

# Caminho para os dados brutos e saída dos dados processados
RAW_DATA_PATH = "app/Fake.br-Corpus/preprocessed/pre-processed-merged.csv"
OUTPUT_DIR = "app/data"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# 1. Carregar os dados
df = pd.read_csv(RAW_DATA_PATH)

# 2. Normalizar e mapear labels
df['label'] = df['label'].astype(str).str.strip().str.lower()
df['label'] = df['label'].map({'fake': 1, 'true': 0})

# Diagnóstico de rótulos inválidos
if df['label'].isnull().any():
    print("[ERRO] Existem rótulos não reconhecidos:")
    print(df[df['label'].isnull()][['label', 'preprocessed_news']].head())
    exit(1)


# 3. Separar features e labels
X = df['preprocessed_news']
y = df['label']

# 4. Separar treino e teste
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 5. Criar o TF-IDF vectorizer (com n-gramas para capturar frases curtas)
vectorizer = TfidfVectorizer(ngram_range=(1, 3), min_df=2, max_df=0.9)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# 6. Salvar os vetores e o vetorizer
joblib.dump(vectorizer, os.path.join(OUTPUT_DIR, "tfidf_vectorizer.joblib"))
np.savez_compressed(os.path.join(OUTPUT_DIR, "train_data.npz"), 
                    X=X_train_tfidf, y=y_train.values)
np.savez_compressed(os.path.join(OUTPUT_DIR, "test_data.npz"), 
                    X=X_test_tfidf, y=y_test.values)

print("[OK] Pré-processamento concluído e arquivos salvos em app/data/")
