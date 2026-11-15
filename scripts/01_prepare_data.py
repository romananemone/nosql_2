# scripts/01_prepare_data.py
# Запуск з кореня репозиторію: python scripts/01_prepare_data.py
#
# Перші 10k рядків файлу це лише квітень-травень 2007 і майже жодної cs.*
# статті (файл відсортований за arXiv ID). Тому читаємо ВЕСЬ файл і робимо
# випадкову вибірку (reservoir sampling) з фіксованим seed у двох "кошиках":
#   - cs.*    (щоб ML-запити з частин 3-5 мали що знаходити)
#   - решта   (щоб фільтр "будь-яка категорія" мав сенс)
# Випадкова вибірка по всьому файлу автоматично дає розкид за роками.
import json
import os
import random
import pandas as pd
from tqdm import tqdm

INPUT_FILE  = "arxiv-metadata-oai-snapshot.json"
OUTPUT_FILE = "data/arxiv_subset.parquet"
MAX_RECORDS = 10_000
CS_SHARE    = 0.6          # частка статей з основною категорією cs.*
SEED        = 42

os.makedirs("data", exist_ok=True)


def clean_text(s: str) -> str:
    """Прибираємо переноси рядків і зайві пробіли."""
    return " ".join(s.split())


def extract_year(paper: dict) -> int:
    """
    Беремо рік із першої версії статті, це дата публікації на arXiv.
    update_date це дата останнього оновлення, вона може бути на роки пізніше.
    Формат created: "Mon, 2 Apr 2007 19:18:42 GMT"
    """
    try:
        versions = paper.get("versions", [])
        if versions:
            created = versions[0]["created"]
            return int(created.split()[3])
    except (IndexError, ValueError, KeyError):
        pass
    return int(paper.get("update_date", "2000-01-01")[:4])


def format_authors(paper: dict) -> str:
    """
    authors_parsed: [["Прізвище", "Ініціали", ""]] -> "Прізвище І., Прізвище І."
    Якщо authors_parsed немає, беремо сирий рядок authors.
    """
    parsed = paper.get("authors_parsed", [])
    if parsed:
        parts = []
        for entry in parsed[:10]:  # не більше 10 авторів
            last  = entry[0].strip() if len(entry) > 0 else ""
            first = entry[1].strip() if len(entry) > 1 else ""
            if last:
                parts.append(f"{last} {first}".strip())
        return ", ".join(parts)
    return clean_text(paper.get("authors", ""))


def to_record(paper: dict, primary_category: str) -> dict:
    return {
        "id":       paper["id"],
        "title":    clean_text(paper["title"]),
        "abstract": clean_text(paper["abstract"]),
        "authors":  format_authors(paper),
        "year":     extract_year(paper),
        "category": primary_category,
    }


rng = random.Random(SEED)
targets = {"cs": int(MAX_RECORDS * CS_SHARE)}
targets["other"] = MAX_RECORDS - targets["cs"]
reservoirs = {"cs": [], "other": []}
seen = {"cs": 0, "other": 0}

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    for line in tqdm(f, desc="Читаємо датасет", unit=" рядків"):
        line = line.strip()
        if not line:
            continue
        paper = json.loads(line)

        # Пропускаємо записи без анотації або заголовка
        if not (paper.get("abstract") or "").strip() or not (paper.get("title") or "").strip():
            continue

        # categories може містити кілька категорій через пробіл: "cs.LG cs.AI"
        # Беремо першу як основну
        primary_category = (paper.get("categories") or "unknown").split()[0]
        bucket = "cs" if primary_category.startswith("cs.") else "other"

        # Reservoir sampling (алгоритм R): рівноймовірна вибірка з потоку
        seen[bucket] += 1
        res, k = reservoirs[bucket], targets[bucket]
        if len(res) < k:
            res.append(to_record(paper, primary_category))
        else:
            j = rng.randrange(seen[bucket])
            if j < k:
                res[j] = to_record(paper, primary_category)

records = reservoirs["cs"] + reservoirs["other"]
df = pd.DataFrame(records).sort_values("id").reset_index(drop=True)

print(f"\nПереглянуто статей: cs.* = {seen['cs']}, інші = {seen['other']}")
print(f"Завантажено статей: {len(df)}")
print("\nРозподіл за категоріями (топ-10):")
print(df["category"].value_counts().head(10))
print("\nРозподіл за роками:")
print(df["year"].value_counts().sort_index())
print("\nПриклад запису:")
print(df.iloc[0].to_dict())

df.to_parquet(OUTPUT_FILE, index=False)
print(f"\nЗбережено в {OUTPUT_FILE}")
