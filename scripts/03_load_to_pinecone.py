# scripts/03_load_to_pinecone.py
# Запуск з кореня репозиторію: python scripts/03_load_to_pinecone.py
import os
import time
import numpy as np
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

INPUT_PARQUET = "data/arxiv_subset.parquet"
INPUT_EMBEDDINGS = "embeddings/embeddings.npy"
INDEX_NAME = "arxiv-papers"
VECTOR_DIM = 768
METRIC = "cosine"   # specter2 + normalize_embeddings=True -> cosine (== dot product)
BATCH_SIZE = 200    # Pinecone рекомендує батчі до 200 векторів

# Ініціалізація клієнта
pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])

# 1. Створюємо індекс (якщо не існує) і підключаємося до нього
if INDEX_NAME not in pc.list_indexes().names():
    print(f"Створюємо індекс '{INDEX_NAME}' (dim={VECTOR_DIM}, metric={METRIC})...")
    pc.create_index(
        name=INDEX_NAME,
        dimension=VECTOR_DIM,
        metric=METRIC,
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),  # регіон Starter-тіру
    )
    while not pc.describe_index(INDEX_NAME).status["ready"]:
        time.sleep(1)
else:
    print(f"Індекс '{INDEX_NAME}' вже існує, використовуємо його")

index = pc.Index(INDEX_NAME)

# 2. Завантажуємо дані
df = pd.read_parquet(INPUT_PARQUET)
embeddings = np.load(INPUT_EMBEDDINGS)
assert len(df) == len(embeddings), "Кількість рядків parquet і ембеддингів не збігається"
assert embeddings.shape[1] == VECTOR_DIM, f"Очікувана розмірність {VECTOR_DIM}"
print(f"Статей: {len(df)}, ембеддинги: {embeddings.shape}")


def build_vector(idx: int, row) -> dict:
    """Вектор для Pinecone: id вигляду paper_<номер рядка> + ембеддинг + метадані.
    Номер рядка збігається з індексом у parquet і embeddings.npy, тож повний
    abstract після пошуку можна взяти як df.iloc[idx]."""
    return {
        "id": f"paper_{idx}",
        "values": embeddings[idx].tolist(),
        "metadata": {
            "arxiv_id": row.id,
            "title":    row.title,
            "abstract": row.abstract[:500],  # ліміт метаданих Pinecone 40 KB на вектор
            "authors":  row.authors[:200],
            "year":     int(row.year),
            "category": row.category,
        },
    }


# 3-4. Формуємо батчі і завантажуємо з прогресом
for start in tqdm(range(0, len(df), BATCH_SIZE), desc="Upsert у Pinecone", unit="батч"):
    batch = df.iloc[start:start + BATCH_SIZE]
    vectors = [
        build_vector(start + pos, row)
        for pos, row in enumerate(batch.itertuples(index=False))
    ]
    index.upsert(vectors=vectors)

# 5. Статистика індексу (лічильник оновлюється з невеликою затримкою)
for _ in range(30):
    stats = index.describe_index_stats()
    if stats.total_vector_count >= len(df):
        break
    time.sleep(2)

print(f"\nВекторів в індексі: {stats.total_vector_count}")
print(f"Розмірність: {stats.dimension}")
print(stats)
