# scripts/05_chunking.py
# Запуск з кореня репозиторію: python scripts/05_chunking.py
import os
import re
import time
import numpy as np
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer

load_dotenv()

MODEL_NAME = "allenai/specter2_base"
VECTOR_DIM = 768

N_PAPERS = 30            # беремо статті з найдовшими анотаціями
FIXED_SIZE = 50          # слів у fixed-size чанку
FIXED_OVERLAP = 10       # слів перекриття між сусідніми fixed-size чанками
SEMANTIC_MAX_WORDS = 60  # максимум слів у semantic чанку
BATCH_SIZE = 100
TOP_K = 5

INDEXES = {
    "fixed": "arxiv-chunks-fixed",
    "semantic": "arxiv-chunks-semantic",
}

pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
model = SentenceTransformer(MODEL_NAME)
df = pd.read_parquet("data/arxiv_subset.parquet")


# 1. 30 статей з найдовшими анотаціями
df["n_words"] = df["abstract"].str.split().str.len()
papers = df.nlargest(N_PAPERS, "n_words")
print(f"Обрано {len(papers)} статей, довжина анотацій: "
      f"{papers.n_words.min()}-{papers.n_words.max()} слів")


# 2. Стратегії chunking
def fixed_size_chunks(text: str, size: int = FIXED_SIZE, overlap: int = FIXED_OVERLAP) -> list[str]:
    """Вікно з `size` слів, яке зсувається на size - overlap слів.
    Межі речень ігноруються, тому речення може бути розрізане."""
    words = text.split()
    step = size - overlap
    chunks = []
    for start in range(0, len(words), step):
        chunks.append(" ".join(words[start:start + size]))
        if start + size >= len(words):
            break
    return chunks


# Кінець речення: . ! ? і далі велика літера / цифра / дужка / $ (LaTeX)
SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(\[$])")


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in SENTENCE_END.split(text) if s.strip()]


def semantic_chunks(text: str, max_words: int = SEMANTIC_MAX_WORDS) -> list[str]:
    """Додаємо речення в чанк, поки не перевищимо max_words.
    Речення ніколи не розрізається; задовге речення стає окремим чанком."""
    chunks, current, current_len = [], [], 0
    for sent in split_sentences(text):
        n = len(sent.split())
        if current and current_len + n > max_words:
            chunks.append(" ".join(current))
            current, current_len = [], 0
        current.append(sent)
        current_len += n
    if current:
        chunks.append(" ".join(current))
    return chunks


def ends_mid_sentence(chunk: str) -> bool:
    return not chunk.rstrip().endswith((".", "!", "?"))


strategies = {"fixed": fixed_size_chunks, "semantic": semantic_chunks}
all_chunks = {name: [] for name in strategies}

for idx, paper in papers.iterrows():
    for name, chunker in strategies.items():
        for n, text in enumerate(chunker(paper.abstract)):
            all_chunks[name].append({
                "id": f"paper_{idx}_chunk_{n}",
                "text": text,
                "metadata": {
                    "arxiv_id": paper.id,
                    "title": paper.title,
                    "text": text,
                    "chunk_no": n,
                    "year": int(paper.year),
                    "category": paper.category,
                },
            })

print("\nСтатистика чанків:")
for name, chunks in all_chunks.items():
    lengths = [len(c["text"].split()) for c in chunks]
    cut = sum(ends_mid_sentence(c["text"]) for c in chunks)
    print(f"  {name:9s} чанків: {len(chunks):4d} | слів: середнє {np.mean(lengths):5.1f}, "
          f"min {min(lengths)}, max {max(lengths)} | закінчуються посеред речення: {cut}")

example = papers.iloc[0]
print(f"\nПриклад для статті: {example.title}")
for name, chunker in strategies.items():
    print(f"\n  [{name}]")
    for n, text in enumerate(chunker(example.abstract)[:3]):
        print(f"   #{n}: {text}")

# Вплив overlap на кількість чанків і покриття тексту
total_words = int(papers.n_words.sum())
print(f"\nВплив overlap (fixed, size={FIXED_SIZE}), всього слів у текстах: {total_words}")
for ov in (0, 10, 20, 30):
    chunks = [c for t in papers.abstract for c in fixed_size_chunks(t, FIXED_SIZE, ov)]
    words_in_chunks = sum(len(c.split()) for c in chunks)
    print(f"  overlap={ov:2d}: чанків {len(chunks):4d}, слів у чанках {words_in_chunks:5d} "
          f"(x{words_in_chunks / total_words:.2f} від тексту)")


# 3. Індекси в Pinecone
def get_index(name: str):
    if name not in pc.list_indexes().names():
        print(f"Створюємо індекс '{name}'...")
        pc.create_index(
            name=name,
            dimension=VECTOR_DIM,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        while not pc.describe_index(name).status["ready"]:
            time.sleep(1)
    index = pc.Index(name)
    # Чистимо старі чанки, щоб після зміни параметрів не лишилось зайвих
    if index.describe_index_stats().total_vector_count > 0:
        index.delete(delete_all=True)
    return index


indexes = {name: get_index(index_name) for name, index_name in INDEXES.items()}

# 4-5. Ембеддинги чанків і завантаження батчами
for name, chunks in all_chunks.items():
    print(f"\n[{name}] кодуємо {len(chunks)} чанків")
    vectors = model.encode(
        [c["text"] for c in chunks],
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    for start in tqdm(range(0, len(chunks), BATCH_SIZE), desc=f"Upsert {INDEXES[name]}", unit="батч"):
        batch = chunks[start:start + BATCH_SIZE]
        indexes[name].upsert(vectors=[
            {"id": c["id"], "values": vectors[start + i].tolist(), "metadata": c["metadata"]}
            for i, c in enumerate(batch)
        ])

for name, index in indexes.items():
    expected = len(all_chunks[name])
    for _ in range(30):
        count = index.describe_index_stats().total_vector_count
        if count >= expected:
            break
        time.sleep(2)
    print(f"Векторів в '{INDEXES[name]}': {count} (очікувалось {expected})")


# 6. Пошук по чанках
def search_chunks(query: str, strategy: str, top_k: int = TOP_K):
    q = model.encode(query, normalize_embeddings=True)
    return indexes[strategy].query(vector=q.tolist(), top_k=top_k, include_metadata=True).matches


queries = [
    "searching for habitable Earth-like planets around other stars",
    "predicting software bugs with recurrent neural networks",
    "fast sorting algorithms for small arrays",
    "structure of neutron stars and their equations of state",
]

for query in queries:
    print(f"\n\nЗапит: '{query}'")
    for strategy in strategies:
        print(f"\n  [{strategy}]")
        for rank, m in enumerate(search_chunks(query, strategy), 1):
            md = m.metadata
            print(f"  {rank}. [{m.score:.4f}] {md['title'][:80]} (чанк #{int(md['chunk_no'])})")
            print(f"     {md['text'][:160]}...")
