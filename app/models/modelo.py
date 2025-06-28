import pandas as pd
import numpy as np
import string
import joblib
import nltk
from textblob import TextBlob
from textblob.en.sentiments import PatternAnalyzer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import FunctionTransformer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.base import BaseEstimator, TransformerMixin
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

from app.utils.transformers import select_text_column, select_sentiment_column




print("🔧 Baixando recursos do NLTK...")
nltk.download('stopwords')

print("📥 Lendo arquivos CSV...")
true_df = pd.read_csv("app/data/True.csv")
fake_df = pd.read_csv("app/data/Fake.csv")

print("🏷️ Adicionando rótulos...")
true_df["label"] = 1
fake_df["label"] = 0

print("📊 Concatenando os dados...")
df = pd.concat([true_df, fake_df], ignore_index=True)
df = df[["text", "label"]].dropna()

print("🧹 Iniciando pré-processamento de texto...")
stemmer = PorterStemmer()
stop_words = set(stopwords.words("english"))

def preprocess_text(text):
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    tokens = text.split()
    tokens = [stemmer.stem(word) for word in tokens if word not in stop_words]
    return " ".join(tokens)

print("🔁 Aplicando pré-processamento...")
df["text"] = df["text"].apply(preprocess_text)

print("🧠 Calculando sentimentos com TextBlob...")

analisador = PatternAnalyzer()

def analisar_sentimento(text: str) -> float:
    return float(analisador.analyze(text).polarity)

df["sentimento"] = df["text"].apply(analisar_sentimento)

# Define variáveis
X = df[["text", "sentimento"]]
y = df["label"]

print("✂️ Separando treino, validação e teste...")
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.1765, random_state=42)  # ≈15%

print("⚙️ Montando pipeline com TF-IDF + Sentimento...")




text_transformer = Pipeline([
    ("selector", FunctionTransformer(select_text_column, validate=False)),
    ("tfidf", TfidfVectorizer(max_features=5000))
])

sentiment_transformer = Pipeline([
    ("selector", FunctionTransformer(select_sentiment_column, validate=False))
])

# Combina texto + sentimento
feature_union = FeatureUnion([
    ("text", text_transformer),
    ("sentimento", sentiment_transformer)
])

# Pipeline completo
pipeline = Pipeline([
    ("features", feature_union),
    ("clf", LogisticRegression())
])

print("🧠 Treinando o modelo...")
pipeline.fit(X_train, y_train)

print("📈 Avaliando o modelo...")
train_acc = accuracy_score(y_train, pipeline.predict(X_train))
val_acc = accuracy_score(y_val, pipeline.predict(X_val))
test_acc = accuracy_score(y_test, pipeline.predict(X_test))

print("\n✅ Avaliação do modelo:")
print(f"Treino: {train_acc:.2f}")
print(f"Validação: {val_acc:.2f}")
print(f"Teste: {test_acc:.2f}")

print("\n📊 Relatório de Classificação (Teste):")
print(classification_report(y_test, pipeline.predict(X_test)))

print("\n📉 Matriz de Confusão:")
print(confusion_matrix(y_test, pipeline.predict(X_test)))

print("💾 Salvando modelo treinado com sentimento...")
joblib.dump(pipeline, "app/models/modelo_fake_news.joblib")

print("\n🚀 Treinamento finalizado e modelo salvo com sucesso!")
