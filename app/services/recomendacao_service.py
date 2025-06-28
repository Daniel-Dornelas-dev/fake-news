import string
from textblob.en.sentiments import PatternAnalyzer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from app.utils.carregador_modelo import carregar_modelo
from deep_translator import GoogleTranslator

# Pré-processadores
stemmer = PorterStemmer()
stop_words = set(stopwords.words("english"))
analisador = PatternAnalyzer()
modelo = carregar_modelo()

def traduzir_para_ingles(texto: str) -> str:
    try:
        return GoogleTranslator(source='auto', target='en').translate(texto)
    except Exception as e:
        print(f"⚠️ Erro ao traduzir o texto: {e}")
        return texto  # Fallback: retorna o texto original se a tradução falhar

def preprocessar_texto(texto: str) -> str:
    texto = texto.lower()
    texto = texto.translate(str.maketrans("", "", string.punctuation))
    tokens = texto.split()
    tokens = [stemmer.stem(word) for word in tokens if word not in stop_words]
    return " ".join(tokens)

def analisar_sentimento(texto: str) -> float:
    return float(analisador.analyze(texto).polarity)

def classificar(texto: str):
    texto_traduzido = traduzir_para_ingles(texto)
    texto_preprocessado = preprocessar_texto(texto_traduzido)
    sentimento = analisar_sentimento(texto_preprocessado)

    entrada = {
        "text": [texto_preprocessado],
        "sentimento": [sentimento]
    }

    pred = modelo.predict(entrada)[0]
    prob = modelo.predict_proba(entrada)[0][1 if pred == 1 else 0]

    return {
        "texto": texto,
        "fake": bool(pred == 0),
        "probabilidade": round(float(prob), 4)
    }
