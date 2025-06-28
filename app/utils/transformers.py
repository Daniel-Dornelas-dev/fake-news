# app/utils/transformers.py

def select_text_column(dataframe):
    return dataframe["text"]

def select_sentiment_column(dataframe):
    return dataframe["sentimento"].values.reshape(-1, 1)
