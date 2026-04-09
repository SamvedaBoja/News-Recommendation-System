# src/live_store.py

import sqlite3
import numpy as np
import json
import os
import spacy
from datetime import datetime, timezone
from sentence_transformers import SentenceTransformer

DB_PATH   = "live_news.db"
BGE_MODEL = SentenceTransformer("BAAI/bge-base-en-v1.5")

_nlp = None

def _get_nlp():
    global _nlp
    if _nlp is None:
        _nlp = spacy.load("en_core_web_sm")
    return _nlp


def extract_entities(text: str) -> list:
    nlp = _get_nlp()
    doc = nlp(text)
    entities = []
    seen = set()
    for ent in doc.ents:
        if ent.label_ in ("PERSON", "ORG", "GPE", "EVENT", "PRODUCT"):
            key = (ent.text.lower(), ent.label_)
            if key not in seen:
                seen.add(key)
                entities.append({"label": ent.text, "type": ent.label_})
    return entities


def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS live_articles (
            news_id      TEXT PRIMARY KEY,
            title        TEXT,
            abstract     TEXT,
            category     TEXT,
            source       TEXT,
            url          TEXT,
            published_at TEXT,
            embedding    BLOB,
            trend_score  REAL,
            fetched_at   TEXT,
            entities     TEXT
        )
    """)
    # Migrate older DBs that may not have the entities column
    try:
        conn.execute("ALTER TABLE live_articles ADD COLUMN entities TEXT")
    except sqlite3.OperationalError:
        pass  # column already exists
    conn.commit()
    conn.close()


def encode_and_store(articles: list):
    """BGE encode + entity extraction, then save to SQLite."""
    conn  = sqlite3.connect(DB_PATH)
    now   = datetime.now(timezone.utc).isoformat()
    texts = [f"{a['title']} {a['abstract']}" for a in articles]

    embeddings = BGE_MODEL.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=False
    )

    for i, a in enumerate(articles):
        trend    = compute_live_trend(a["published_at"])
        entities = extract_entities(f"{a['title']} {a['abstract']}")
        conn.execute("""
            INSERT OR REPLACE INTO live_articles
            VALUES (?,?,?,?,?,?,?,?,?,?,?)
        """, (
            a["news_id"], a["title"], a["abstract"],
            a["category"], a["source"], a["url"],
            a["published_at"],
            embeddings[i].astype(np.float32).tobytes(),
            trend, now,
            json.dumps(entities),
        ))
    conn.commit()
    conn.close()


def compute_live_trend(published_at: str) -> float:
    """Real time-decay using actual publish timestamp.
    Decay rate 0.05 gives ~14-hour half-life, keeping day-old articles visible.
      0h → 1.00,  6h → 0.74,  12h → 0.55,  24h → 0.30,  48h → 0.09
    """
    try:
        pub_time  = datetime.fromisoformat(published_at.replace("Z", "+00:00"))
        now       = datetime.now(timezone.utc)
        hours_old = (now - pub_time).total_seconds() / 3600
        return float(np.exp(-0.05 * hours_old))
    except Exception:
        return 0.5


def load_live_articles() -> tuple:
    """Returns (articles_list, embeddings_matrix)"""
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute("""
        SELECT news_id, title, abstract, category,
               source, url, published_at, embedding, trend_score,
               fetched_at, entities
        FROM live_articles
        ORDER BY trend_score DESC
    """).fetchall()
    conn.close()

    articles   = []
    embeddings = []
    for row in rows:
        try:
            entities = json.loads(row[10]) if row[10] else []
        except (json.JSONDecodeError, TypeError):
            entities = []

        articles.append({
            "news_id"     : row[0],
            "title"       : row[1],
            "abstract"    : row[2],
            "category"    : row[3],
            "source"      : row[4],
            "url"         : row[5],
            "published_at": row[6],
            "trend_score" : compute_live_trend(row[6]),  # recompute from timestamp, not stale DB value
            "fetched_at"  : row[9] or "",
            "entities"    : entities,
        })
        vec = np.frombuffer(row[7], dtype=np.float32)
        embeddings.append(vec)

    emb_matrix = np.array(embeddings) if embeddings else np.zeros((0, 768))
    return articles, emb_matrix
