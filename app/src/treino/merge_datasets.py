import pandas as pd
import os

# Caminhos
DATA_DIR = "app/Fake.br-Corpus/preprocessed"
BASE_ORIGINAL = os.path.join(DATA_DIR, "pre-processed.csv")
BASE_SOCIAIS = os.path.join(DATA_DIR, "frases_sociais_positivas_500.csv")
BASE_CURTAS = os.path.join(DATA_DIR, "frases_curtas_augmentadas_600.csv")
BASE_MERGED = os.path.join(DATA_DIR, "pre-processed-merged.csv")

# 1. Carregar os três datasets
df_original = pd.read_csv(BASE_ORIGINAL)
df_sociais = pd.read_csv(BASE_SOCIAIS)
df_curtas = pd.read_csv(BASE_CURTAS)

# 2. Concatenar todos
df_merged = pd.concat([df_original, df_sociais, df_curtas], ignore_index=True)

# 3. Reajustar o índice
df_merged['index'] = range(len(df_merged))

# 4. Salvar o novo dataset consolidado
df_merged.to_csv(BASE_MERGED, index=False)

print(f"[OK] Dataset unificado salvo em: {BASE_MERGED}")
print(f"Total de registros: {len(df_merged)}")
