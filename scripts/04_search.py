# scripts/04_search.py
# Запуск з кореня репозиторію: python scripts/04_search.py
import os
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer

load_dotenv()

INDEX_NAME = "arxiv-papers"
MODEL_NAME = "allenai/specter2_base"
TOP_K = 5
ABSTRACT_PREVIEW = 200   # скільки символів абстракту показувати

pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
index = pc.Index(INDEX_NAME)
model = SentenceTransformer(MODEL_NAME)
df = pd.read_parquet("data/arxiv_subset.parquet")  # для отримання повного abstract


# Допоміжні функції
def encode_query(text: str) -> np.ndarray:
    """Кодуємо запит тією ж моделлю і так само нормалізуємо, як і статті."""
    return model.encode(text, normalize_embeddings=True, convert_to_numpy=True)


def row_of(vector_id: str) -> pd.Series:
    """paper_<i> -> i-й рядок parquet (там повний abstract, а не обрізаний до 500)."""
    return df.iloc[int(vector_id.split("_")[1])]


def search(query_vec: np.ndarray, flt: dict | None = None, top_k: int = TOP_K):
    return index.query(
        vector=query_vec.tolist(),
        top_k=top_k,
        include_metadata=True,
        filter=flt,
    ).matches


def print_matches(title: str, matches) -> None:
    print(f"\n\n{title}")
    if not matches:
        print("  (нічого не знайдено)")
        return
    for rank, m in enumerate(matches, 1):
        paper = row_of(m.id)
        print(f"{rank}. [{m.score:.4f}] {paper.title}")
        print(f"   {paper.category} | {paper.year} | arXiv:{paper.id}")
        print(f"   {paper.abstract[:ABSTRACT_PREVIEW]}...")


def print_local_top(title: str, scores: np.ndarray, top_idx: np.ndarray) -> None:
    print(f"\n{title}:")
    for rank, i in enumerate(top_idx, 1):
        print(f"{rank}. [{scores[i]:.6f}] {df.iloc[i].title[:80]}")


# 1. Чистий семантичний пошук
query = "teaching machines to recognize objects in pictures"
q_vec = encode_query(query)
print(f"Норма вектора запиту: {np.linalg.norm(q_vec):.6f}")

pure = search(q_vec)
print_matches(f"Семантичний пошук: '{query}'", pure)

# 2. Пошук з фільтрами по метаданих
max_year = int(df["year"].max())
recent_from = max_year - 4          # "останні 5 років": max_year-4 ... max_year

# A: reinforcement learning, тільки cs.LG, останні 5 років
query_rl = "agents learning to act through trial and error and rewards"
q_rl = encode_query(query_rl)
filter_a = {"category": {"$eq": "cs.LG"}, "year": {"$gte": recent_from}}
res_a = search(q_rl, filter_a)
print_matches(f"A: '{query_rl}' | category = cs.LG, year >= {recent_from}", res_a)

# B: той самий запит, старі статті (до 2015), будь-яка категорія
filter_b = {"year": {"$lt": 2015}}
res_b = search(q_rl, filter_b)
print_matches(f"B: '{query_rl}' | year < 2015, будь-яка категорія", res_b)

# Для порівняння: той самий запит без фільтрів
res_nf = search(q_rl)
print_matches(f"Без фільтрів: '{query_rl}'", res_nf)

ids_a, ids_b, ids_nf = ({m.id for m in r} for r in (res_a, res_b, res_nf))
print("\nПорівняння видачі:")
print(f"  спільних A ∩ B:            {len(ids_a & ids_b)}")
print(f"  спільних A ∩ без фільтра:  {len(ids_a & ids_nf)}")
print(f"  спільних B ∩ без фільтра:  {len(ids_b & ids_nf)}")
print(f"  категорії в B: {sorted({row_of(m.id).category for m in res_b})}")
print(f"  роки в A: {sorted(int(row_of(m.id).year) for m in res_a)}")
print(f"  роки в B: {sorted(int(row_of(m.id).year) for m in res_b)}")

# 3. Порівняння метрик на локальних ембеддингах
embeddings = np.load("embeddings/embeddings.npy")
print(f"\nЛокальні ембеддинги: {embeddings.shape}, "
      f"норми від {np.linalg.norm(embeddings, axis=1).min():.4f} "
      f"до {np.linalg.norm(embeddings, axis=1).max():.4f}")

cosine = embeddings @ q_vec / (np.linalg.norm(embeddings, axis=1) * np.linalg.norm(q_vec))
dot = embeddings @ q_vec
l2 = np.linalg.norm(embeddings - q_vec, axis=1)

top_cos = np.argsort(-cosine)[:TOP_K]   # більше = схожіше
top_dot = np.argsort(-dot)[:TOP_K]      # більше = схожіше
top_l2 = np.argsort(l2)[:TOP_K]         # менше = схожіше

print(f"\nЗапит: '{query}'")
print_local_top("Cosine similarity (більше = краще)", cosine, top_cos)
print_local_top("Dot product (більше = краще)", dot, top_dot)
print_local_top("L2 distance (менше = краще)", l2, top_l2)

print("\nЧи збігаються топ-5:")
print(f"  cosine == dot: {np.array_equal(top_cos, top_dot)}")
print(f"  cosine == L2:  {np.array_equal(top_cos, top_l2)}")
# Для одиничних векторів: ||a - b||^2 = 2 - 2 * cos(a, b)
i = top_cos[0]
print(f"  перевірка L2^2 = 2 - 2*cos для топ-1: "
      f"{l2[i] ** 2:.6f} vs {2 - 2 * cosine[i]:.6f}")

# Збіг із Pinecone (індекс створено з metric=cosine)
pinecone_ids = [int(m.id.split("_")[1]) for m in pure]
print(f"  Pinecone == локальний cosine: {pinecone_ids == top_cos.tolist()}")
