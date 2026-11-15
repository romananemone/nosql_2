# scripts/02_embed.py
# Запуск з кореня репозиторію: python scripts/02_embed.py
import os
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

INPUT_PARQUET = "data/arxiv_subset.parquet"
OUTPUT_FILE = "embeddings/embeddings.npy"
MODEL_NAME = "allenai/specter2_base"
BATCH_SIZE = 64

# 1. Завантаження датасету
df = pd.read_parquet(INPUT_PARQUET)

# 2. Формат, на якому навчена модель: title [SEP] abstract
texts = (df["title"] + " [SEP] " + df["abstract"]).tolist()

# 3. Модель. specter2_base не має sentence-transformers конфігу,
#    тому бібліотека сама додає mean pooling поверх BERT.
model = SentenceTransformer(MODEL_NAME)

# 4. Кодування: батчі, прогрес, L2-нормалізація (вектори одиничної довжини)
embeddings = model.encode(
    texts,
    batch_size=BATCH_SIZE,
    show_progress_bar=True,
    normalize_embeddings=True,
    convert_to_numpy=True,
)

# 5. Перевірка
print(f"Кількість текстів:      {len(texts)}")
print(f"Розмірність ембеддингу: {embeddings.shape[1]}")
print(f"Норма першого вектора:  {np.linalg.norm(embeddings[0]):.6f}")

# 6-7. Збереження (рядок i ембеддингів відповідає рядку i parquet)
os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
np.save(OUTPUT_FILE, embeddings.astype(np.float32))
print(f"Збережено в {OUTPUT_FILE}")
