import pandas as pd
import numpy as np
import json
import os
import requests
from datetime import datetime, timezone
import math
from dotenv import load_dotenv

# Load .env file
load_dotenv()

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED = os.path.join(BASE_DIR, "data", "processed")
CACHE_DIR = os.path.join(BASE_DIR, "data", "cache")
os.makedirs(CACHE_DIR, exist_ok=True)

# ── Constants ──────────────────────────────────────────────────────────────
LAMBDA         = 0.05   # decay rate: score halves every ~14 hours
TREND_WINDOW   = 24     # hours — articles published in last 24h get trend boost
NEWSAPI_KEY    = os.getenv("NEWSAPI_KEY", "")   # replace with your key
CACHE_FILE     = os.path.join(CACHE_DIR, "newsapi_cache.json")
CACHE_DURATION = 1800   # seconds — refresh cache every 30 minutes

if __name__ == "__main__":
    # Verify API key loaded correctly
    key_preview = NEWSAPI_KEY[:8] + "..." if NEWSAPI_KEY else "NOT FOUND"
    print(f"NewsAPI key loaded: {key_preview}")

# ── Step 1: Time-Decay Score ───────────────────────────────────────────────
def compute_time_decay(publish_time_str, lambda_val=LAMBDA):
    """
    Compute freshness score using exponential decay.
    
    Score = e^(-lambda * delta_hours)
    
    Fresh article (0 hours old)  → score ≈ 1.0
    6 hours old                  → score ≈ 0.74
    14 hours old                 → score ≈ 0.50
    24 hours old                 → score ≈ 0.30
    72 hours old                 → score ≈ 0.027
    """
    try:
        # MIND timestamps format: "11/15/2019 9:05:40 AM"
        pub_time = datetime.strptime(
            str(publish_time_str).strip(), 
            "%m/%d/%Y %I:%M:%S %p"
        )
        pub_time = pub_time.replace(tzinfo=timezone.utc)
        now      = datetime.now(timezone.utc)
        delta_hours = (now - pub_time).total_seconds() / 3600
        score = math.exp(-lambda_val * delta_hours)
        return round(score, 6)
    except (ValueError, TypeError):
        # If timestamp is missing or malformed, return a neutral score
        return 0.1


def compute_time_decay_batch(news_df, lambda_val=LAMBDA):
    """Add time_decay_score column to news_df."""
    news_df = news_df.copy()
    
    # For MIND dataset: timestamps exist in behaviors but not in news directly
    # We simulate recency using article index as proxy
    # (lower index = older in MIND-small which covers Nov 2019)
    # This is academically valid for offline evaluation
    n = len(news_df)
    
    # Simulate: spread articles over a 7-day window
    # Article 0 = 7 days old, last article = just published
    simulated_hours = np.linspace(168, 0, n)   # 168 hours = 7 days
    
    news_df["simulated_age_hours"] = simulated_hours
    news_df["time_decay_score"]    = news_df["simulated_age_hours"].apply(
        lambda h: round(math.exp(-lambda_val * h), 6)
    )
    
    return news_df


# ── Step 2: Entity Trend Detection ────────────────────────────────────────
def compute_entity_trend_scores(news_df, trend_window_hours=TREND_WINDOW):
    """
    Detect trending entities by counting how often each entity
    appears in the most recent articles.
    
    Returns: dict {entity_id: trend_score}
    """
    import ast

    def safe_parse(val):
        if isinstance(val, list):
            return val
        if pd.isna(val) or val == "[]" or val == "":
            return []
        try:
            return ast.literal_eval(val)
        except:
            return []

    # Use the most recent articles (top N by time_decay_score)
    if "time_decay_score" not in news_df.columns:
        news_df = compute_time_decay_batch(news_df)

    # Take top 20% freshest articles as "trending window"
    top_n       = int(len(news_df) * 0.20)
    recent_news = news_df.nlargest(top_n, "time_decay_score")

    entity_counts = {}
    for _, row in recent_news.iterrows():
        entities = safe_parse(row.get("entity_ids", []))
        for eid in entities:
            entity_counts[eid] = entity_counts.get(eid, 0) + 1

    # Normalize counts to 0-1 range
    if not entity_counts:
        return {}
    max_count = max(entity_counts.values())
    entity_trend_scores = {
        eid: round(count / max_count, 4)
        for eid, count in entity_counts.items()
    }
    return entity_trend_scores


# ── Step 3: Personalized Trend Score (Novel Contribution) ─────────────────
def compute_personalized_trend_score(user_id, news_id,
                                      user_interest_profiles,
                                      news_df_lookup,
                                      entity_trend_scores,
                                      time_decay_scores):
    """
    PersonalizedTrendScore = UserCategoryWeight(c) * TimeDecayScore
                           + EntityTrendBoost

    This is the novel improvement over global trend scoring:
    - A sports reader gets sports trends boosted
    - A finance reader gets finance trends boosted
    - Prevents showing earthquake news to someone reading only tech
    """
    # Get user's category weights
    if user_id not in user_interest_profiles:
        cat_weight = 0.5   # neutral for unknown users
    else:
        user_profile = user_interest_profiles[user_id]
        cat_weights  = user_profile.get("category_weights", {})

        # Get article's category
        if news_id not in news_df_lookup:
            return 0.0
        article_cat = str(news_df_lookup[news_id].get("category", "")).lower()
        cat_weight  = cat_weights.get(article_cat, 0.05)  # small default for unseen cats

    # Get time decay score for this article
    time_decay = time_decay_scores.get(news_id, 0.1)

    # Entity trend boost: check if article's entities are currently trending
    import ast
    def safe_parse(val):
        if isinstance(val, list): return val
        try: return ast.literal_eval(str(val))
        except: return []

    if news_id in news_df_lookup:
        article_entities = safe_parse(
            news_df_lookup[news_id].get("entity_ids", [])
        )
        entity_boost = np.mean([
            entity_trend_scores.get(eid, 0.0)
            for eid in article_entities
        ]) if article_entities else 0.0
    else:
        entity_boost = 0.0

    # Final personalized trend score
    # Category weight personalizes the time-decay signal
    # Entity boost adds a trending-topic layer
    personalized_score = (0.6 * cat_weight * time_decay) + (0.4 * entity_boost)
    return round(personalized_score, 4)


# ── Step 4: NewsAPI Integration (Real-Time Layer) ─────────────────────────
def fetch_trending_news(api_key=NEWSAPI_KEY, use_cache=True):
    """
    Fetch latest headlines from NewsAPI.
    Results are cached for 30 minutes to avoid hitting rate limits.
    Returns list of article dicts with title, category, publishedAt.
    """
    # Check cache first
    if use_cache and os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "r") as f:
            cache = json.load(f)
        cache_age = (datetime.now().timestamp() - cache.get("timestamp", 0))
        if cache_age < CACHE_DURATION:
            print(f"✅ Using cached NewsAPI results ({cache_age:.0f}s old)")
            return cache.get("articles", [])

    if api_key == "YOUR_NEWSAPI_KEY_HERE" or not api_key:
        print("⚠️ No NewsAPI key — using simulated trending data")
        return get_simulated_trending_news()

    try:
        url    = "https://newsapi.org/v2/top-headlines"
        params = {
            "language" : "en",
            "pageSize" : 20,
            "apiKey"   : api_key
        }
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data     = response.json()
        articles = data.get("articles", [])

        # Cache results
        with open(CACHE_FILE, "w") as f:
            json.dump({
                "timestamp": datetime.now().timestamp(),
                "articles" : articles
            }, f)

        print(f"✅ Fetched {len(articles)} articles from NewsAPI")
        return articles

    except Exception as e:
        print(f"⚠️ NewsAPI error: {e} — using simulated data")
        return get_simulated_trending_news()


def get_simulated_trending_news():
    """
    Fallback simulated trending news when NewsAPI is unavailable.
    Used for offline testing and demo purposes.
    """
    return [
        {"title": "Breaking: Major tech company announces AI breakthrough",
         "category": "technology", "publishedAt": datetime.now().isoformat()},
        {"title": "Stock markets rally as inflation fears ease",
         "category": "finance",    "publishedAt": datetime.now().isoformat()},
        {"title": "Championship game draws record viewership",
         "category": "sports",     "publishedAt": datetime.now().isoformat()},
        {"title": "New climate agreement signed by world leaders",
         "category": "news",       "publishedAt": datetime.now().isoformat()},
        {"title": "Scientists discover potential cancer treatment",
         "category": "health",     "publishedAt": datetime.now().isoformat()},
    ]


# ── Step 5: Batch Trend Scoring ───────────────────────────────────────────
def get_trend_scores_for_user(user_id, candidate_news_ids,
                               user_interest_profiles,
                               news_df_lookup,
                               entity_trend_scores,
                               time_decay_scores):
    """Compute personalized trend scores for a list of candidate articles."""
    scores = {}
    for nid in candidate_news_ids:
        scores[nid] = compute_personalized_trend_score(
            user_id, nid,
            user_interest_profiles,
            news_df_lookup,
            entity_trend_scores,
            time_decay_scores
        )
    return scores


# ── Main: Build and Test ───────────────────────────────────────────────────
if __name__ == "__main__":
    import ast

    print("Loading data...")
    news_df = pd.read_csv(os.path.join(PROCESSED, "news_enriched.csv"))

    with open(os.path.join(PROCESSED, "user_interest_profiles.json")) as f:
        user_interest_profiles = json.load(f)

    # Build news lookup
    def safe_parse(val):
        if isinstance(val, list): return val
        if pd.isna(val) or val in ("", "[]"): return []
        try: return ast.literal_eval(val)
        except: return []

    news_df_lookup = {
        row["news_id"]: {
            "category"  : row["category"],
            "entity_ids": safe_parse(row["entity_ids"]),
            "title"     : row["title"]
        }
        for _, row in news_df.iterrows()
    }

    # Step 1: Time decay scores
    print("\nComputing time-decay scores...")
    news_df              = compute_time_decay_batch(news_df)
    time_decay_scores    = dict(zip(news_df["news_id"],
                                    news_df["time_decay_score"]))

    print(f"✅ Time-decay scores computed")
    print(f"   Max score (freshest) : {max(time_decay_scores.values()):.4f}")
    print(f"   Min score (oldest)   : {min(time_decay_scores.values()):.6f}")
    print(f"   Sample scores:")
    sample_ids = list(time_decay_scores.keys())[:3]
    for nid in sample_ids:
        age = news_df[news_df["news_id"]==nid]["simulated_age_hours"].values[0]
        print(f"     {nid} — age: {age:.1f}h → decay: {time_decay_scores[nid]}")

    # Step 2: Entity trend scores
    print("\nComputing entity trend scores...")
    entity_trend_scores = compute_entity_trend_scores(news_df)
    top_trending = sorted(entity_trend_scores.items(),
                          key=lambda x: x[1], reverse=True)[:5]
    print(f"✅ Entity trend scores computed")
    print(f"   Total trending entities : {len(entity_trend_scores)}")
    print(f"   Top 5 trending entities : {top_trending}")

    # Step 3: Personalized trend scores
    print("\nTesting personalized trend scores...")
    test_user     = "U13740"
    test_articles = list(news_df["news_id"].head(30))

    trend_scores = get_trend_scores_for_user(
        test_user, test_articles,
        user_interest_profiles, news_df_lookup,
        entity_trend_scores, time_decay_scores
    )

    top_trend = sorted(trend_scores.items(),
                       key=lambda x: x[1], reverse=True)[:5]
    print(f"✅ Personalized trend scores for {test_user}")
    print(f"   Top category : {user_interest_profiles[test_user]['top_category']}")
    print(f"\n   Top 5 trend-scored articles:")
    for nid, score in top_trend:
        title = news_df_lookup[nid]["title"][:60]
        cat   = news_df_lookup[nid]["category"]
        print(f"     [{cat}] {title} → trend: {score}")

    # Step 4: Fetch trending news
    print("\nFetching real-time trending news...")
    trending = fetch_trending_news(use_cache=False)
    print(f"\n📰 Current Trending Headlines:")
    for i, a in enumerate(trending[:5], 1):
        print(f"   {i}. {a['title'][:70]}")

    # Save trend data
    trend_data = {
        "time_decay_scores"  : time_decay_scores,
        "entity_trend_scores": entity_trend_scores
    }
    with open(os.path.join(PROCESSED, "trend_data.json"), "w") as f:
        json.dump(trend_data, f)
    print(f"\n✅ Trend data saved to data/processed/trend_data.json")
