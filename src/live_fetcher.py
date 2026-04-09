# src/live_fetcher.py

import os
import requests
from dotenv import load_dotenv

load_dotenv()

NEWSAPI_KEY = os.getenv("NEWSAPI_KEY", "")
NEWSAPI_URL = "https://newsapi.org/v2/top-headlines"

# Map NewsAPI categories to our MIND categories
CATEGORY_MAP = {
    "business"      : "finance",
    "entertainment" : "entertainment",
    "health"        : "health",
    "science"       : "technology",
    "sports"        : "sports",
    "technology"    : "technology",
    "general"       : "news",
    "politics"      : "news",
}

def fetch_live_articles(category: str = "general",
                        country : str = "us",
                        page_size: int = 20) -> list:
    params = {
        "apiKey"  : NEWSAPI_KEY,
        "category": category,
        "country" : country,
        "pageSize": page_size,
    }
    resp = requests.get(NEWSAPI_URL, params=params, timeout=10)
    resp.raise_for_status()
    articles = resp.json().get("articles", [])

    cleaned = []
    for a in articles:
        title       = a.get("title", "") or ""
        description = a.get("description", "") or ""
        if not title or title == "[Removed]":
            continue
        cleaned.append({
            "news_id"     : f"LIVE_{abs(hash(title)) % 999999}",
            "title"       : title,
            "abstract"    : description,
            "category"    : CATEGORY_MAP.get(category, "news"),
            "published_at": a.get("publishedAt", ""),
            "source"      : a.get("source", {}).get("name", ""),
            "url"         : a.get("url", ""),
        })
    return cleaned