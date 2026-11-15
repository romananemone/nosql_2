# scripts/06_hybrid_search.py  (додаткова частина, не оцінюється)
# Запуск з кореня репозиторію: python scripts/06_hybrid_search.py
import os
import re
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

load_dotenv()

INDEX_NAME = "arxiv-papers"
MODEL_NAME = "allenai/specter2_base"
TOP_K = 10   # беремо ширше, щоб RRF міг переранжувати
SHOW = 5     # скільки результатів показувати
RRF_K = 60

pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
index = pc.Index(INDEX_NAME)
model = SentenceTransformer(MODEL_NAME)
df = pd.read_parquet("data/arxiv_subset.parquet").reset_index(drop=True)


def tokenize(text: str) -> list[str]:
    """Токенізація для BM25: нижній регістр, тільки слова і числа."""
    return re.findall(r"\w+", text.lower())


# 1. BM25-індекс: документ = title + abstract + authors
#    (автори додані, щоб запит з ім'ям автора міг збігтися за токенами)
corpus = (df["title"] + " " + df["abstract"] + " " + df["authors"]).tolist()
bm25 = BM25Okapi([tokenize(doc) for doc in corpus])
print(f"BM25-індекс побудовано: {len(corpus)} документів")


# 2-4. Функції пошуку. Усі повертають номери рядків df, від кращого до гіршого.
def bm25_search(query: str, top_k: int = TOP_K) -> list[int]:
    scores = bm25.get_scores(tokenize(query))
    return np.argsort(-scores)[:top_k].tolist()


def vector_search(query: str, top_k: int = TOP_K) -> list[int]:
    q = model.encode(query, normalize_embeddings=True).tolist()
    res = index.query(vector=q, top_k=top_k)
    return [int(m.id.split("_")[1]) for m in res.matches]   # paper_<i> -> i


def rrf(ranked_lists: list[list[int]], k: int = RRF_K) -> list[tuple[int, float]]:
    """Reciprocal Rank Fusion: score(d) = sum 1 / (k + rank_i(d)), rank з 1.
    Використовує лише позиції, тому несумірні скори BM25 і cosine змішувати не треба."""
    scores: dict[int, float] = {}
    for ranked in ranked_lists:
        for rank, doc in enumerate(ranked, start=1):
            scores[doc] = scores.get(doc, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


def hybrid_search(bm25_res: list[int], vec_res: list[int], k: int = RRF_K):
    return rrf([bm25_res, vec_res], k=k)


def describe(i: int) -> str:
    row = df.iloc[i]
    return f"{row.title[:75]} | {row.category} | {row.year}"


def source(i: int, bm25_res: list[int], vec_res: list[int]) -> str:
    """Позиції документа в топ-10 кожного методу, напр. 'BM25 #7 + Vec #2'."""
    parts = []
    if i in bm25_res:
        parts.append(f"BM25 #{bm25_res.index(i) + 1}")
    if i in vec_res:
        parts.append(f"Vec #{vec_res.index(i) + 1}")
    return " + ".join(parts)


# 5-6. Три запити × три методи
queries = [
    ("точний термін", "BERT fine-tuning"),
    ("ім'я автора", "Yann LeCun convolutional networks"),
    ("перефразування", "making computers understand human emotions from text"),
]

for kind, query in queries:
    bm25_res = bm25_search(query)
    vec_res = vector_search(query)
    hybrid = hybrid_search(bm25_res, vec_res, k=RRF_K)

    print(f"\n\nЗапит ({kind}): '{query}'")

    print(f"\nBM25 топ-{SHOW}:")
    for n, i in enumerate(bm25_res[:SHOW], 1):
        print(f"  {n}. {describe(i)}")

    print(f"\nVector топ-{SHOW}:")
    for n, i in enumerate(vec_res[:SHOW], 1):
        print(f"  {n}. {describe(i)}")

    print(f"\nHybrid RRF (k={RRF_K}) топ-{SHOW}:")
    for n, (i, score) in enumerate(hybrid[:SHOW], 1):
        print(f"  {n}. [{score:.4f}] {describe(i)}  <- {source(i, bm25_res, vec_res)}")

    in_both = [i for i, _ in hybrid[:SHOW] if i in bm25_res and i in vec_res]
    not_in_top5 = [i for i, _ in hybrid[:SHOW] if i not in bm25_res[:SHOW] and i not in vec_res[:SHOW]]
    print(f"\n  у гібридному топ-{SHOW}: знайдено обома методами: {len(in_both)}, "
          f"відсутні в топ-{SHOW} обох методів: {len(not_in_top5)}")

    # Вплив параметра k: k=1 віддає перевагу верхнім позиціям, k=60 згоді методів
    hybrid_k1 = hybrid_search(bm25_res, vec_res, k=1)
    print(f"\n  RRF k=1 топ-{SHOW}:")
    for n, (i, score) in enumerate(hybrid_k1[:SHOW], 1):
        print(f"  {n}. [{score:.4f}] {describe(i)}  <- {source(i, bm25_res, vec_res)}")
    same = [i for i, _ in hybrid_k1[:SHOW]] == [i for i, _ in hybrid[:SHOW]]
    print(f"\n  топ-{SHOW} при k=1 і k={RRF_K} однаковий: {same}")
