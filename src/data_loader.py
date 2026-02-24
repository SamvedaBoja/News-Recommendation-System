import pandas as pd
import numpy as np
import os

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "train")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
os.makedirs(PROCESSED_DIR, exist_ok=True)

# ── Column Definitions ─────────────────────────────────────────────────────
NEWS_COLS = [
    "news_id", "category", "subcategory", "title",
    "abstract", "url", "title_entities", "abstract_entities"
]

BEHAVIOR_COLS = [
    "impression_id", "user_id", "time", "history", "impressions"
]

# ── Loaders ────────────────────────────────────────────────────────────────
def load_news(path=None):
    """Load news.tsv into a clean DataFrame."""
    if path is None:
        path = os.path.join(DATA_DIR, "news.tsv")
    
    df = pd.read_csv(
        path, sep="\t", header=None,
        names=NEWS_COLS, quoting=3  # quoting=3 ignores quote chars, prevents parse errors
    )
    # Fill missing abstracts with empty string
    df["abstract"] = df["abstract"].fillna("")
    # Combine title + abstract into one field for encoding later
    df["text"] = df["title"] + ". " + df["abstract"]
    df["text"] = df["text"].str.strip()
    return df


def load_behaviors(path=None):
    """Load behaviors.tsv into a clean DataFrame."""
    if path is None:
        path = os.path.join(DATA_DIR, "behaviors.tsv")
    
    df = pd.read_csv(
        path, sep="\t", header=None,
        names=BEHAVIOR_COLS
    )
    # Parse click history: "N1 N2 N3" → ["N1", "N2", "N3"]
    df["history"] = df["history"].fillna("").apply(
        lambda x: x.strip().split() if x.strip() else []
    )
    # Parse impressions: "N1-1 N2-0 N3-1" → list of (news_id, clicked) tuples
    df["impressions"] = df["impressions"].apply(parse_impressions)
    return df


def parse_impressions(impression_str):
    """Convert 'N1-1 N2-0' format to list of (news_id, label) tuples."""
    if pd.isna(impression_str) or impression_str == "":
        return []
    result = []
    for item in impression_str.strip().split():
        parts = item.split("-")
        if len(parts) == 2:
            result.append((parts[0], int(parts[1])))
    return result


def load_entity_embeddings(path=None):
    """Load pre-trained TransE entity embeddings from MIND."""
    if path is None:
        path = os.path.join(DATA_DIR, "entity_embedding.vec")
    
    entity_embeddings = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 2:
                continue
            entity_id = parts[0]
            try:
                vector = np.array(parts[1:], dtype=np.float32)
                entity_embeddings[entity_id] = vector
            except ValueError:
                continue  # skip malformed lines
    return entity_embeddings


def build_user_profiles(behaviors_df):
    """
    Build per-user click history dictionary.
    Returns: { user_id: [news_id1, news_id2, ...] }
    """
    user_profiles = {}
    for _, row in behaviors_df.iterrows():
        uid = row["user_id"]
        history = row["history"]
        if uid not in user_profiles:
            user_profiles[uid] = []
        user_profiles[uid].extend(history)
    
    # Deduplicate while preserving order
    user_profiles = {
        uid: list(dict.fromkeys(articles))
        for uid, articles in user_profiles.items()
    }
    return user_profiles


if __name__ == "__main__":
    print("Loading news data...")
    news_df = load_news()
    print(f"  News articles loaded: {len(news_df)}")

    print("Loading behavior data...")
    behaviors_df = load_behaviors()
    print(f"  Behavior records loaded: {len(behaviors_df)}")

    print("Loading entity embeddings...")
    entity_emb = load_entity_embeddings()
    print(f"  Entities with embeddings: {len(entity_emb)}")

    print("Building user profiles...")
    user_profiles = build_user_profiles(behaviors_df)
    print(f"  Unique users: {len(user_profiles)}")

    print("\n── News DataFrame Sample ──")
    print(news_df[["news_id", "category", "title"]].head())

    print("\n── Category Distribution ──")
    print(news_df["category"].value_counts())

    print("\n── Behavior Sample ──")
    print(behaviors_df[["user_id", "time", "history"]].head(3))

    print("\n── User Profile Sample (first user) ──")
    first_user = list(user_profiles.keys())[0]
    print(f"  User: {first_user}")
    print(f"  Clicked articles: {user_profiles[first_user][:5]} ...")

    print("\n── Missing Value Check ──")
    print(news_df[["title", "abstract", "category"]].isnull().sum())
    # Save for use in all future phases
    news_df.to_csv(os.path.join(PROCESSED_DIR, "news_processed.csv"), index=False)
    behaviors_df.to_csv(os.path.join(PROCESSED_DIR, "behaviors_processed.csv"), index=False)
    
    import json
    with open(os.path.join(PROCESSED_DIR, "user_profiles.json"), "w") as f:
        json.dump(user_profiles, f)
    
    print("\n✅ All files saved to data/processed/")
