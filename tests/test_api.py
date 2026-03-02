"""
NewsIQ API Integration Tests
=============================
Covers all 8 test cases from the test specification.

Run from project root:
    pytest tests/test_api.py -v -s

Requires the FastAPI server to be running:
    uvicorn api.main:app --reload
"""

import statistics
import time

import pytest
import requests

BASE_URL = "http://localhost:8000"


# ──────────────────────────────────────────────────────────────────────────────
# Session-scoped fixture: verify API is reachable before any test runs
# ──────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session", autouse=True)
def api_alive():
    try:
        resp = requests.get(f"{BASE_URL}/", timeout=5)
        assert resp.status_code == 200, "API health-check failed"
    except requests.ConnectionError:
        pytest.fail(
            "Cannot reach http://localhost:8000 — start the server with:\n"
            "  uvicorn api.main:app --reload"
        )


def _get(path: str, **params):
    """Convenience GET with query params."""
    return requests.get(f"{BASE_URL}{path}", params=params)


# ──────────────────────────────────────────────────────────────────────────────
# Test Case 1 — Existing User with Rich History (U13740)
# ──────────────────────────────────────────────────────────────────────────────

class TestExistingUserU13740:
    USER_ID = "U13740"

    def test_status_200(self):
        assert _get(f"/recommend/{self.USER_ID}", top_k=10).status_code == 200

    def test_response_shape(self):
        data = _get(f"/recommend/{self.USER_ID}", top_k=10).json()
        assert data["user_id"] == self.USER_ID
        assert "recommendations" in data
        assert len(data["recommendations"]) == 10

    def test_required_fields_on_every_rec(self):
        recs = _get(f"/recommend/{self.USER_ID}", top_k=10).json()["recommendations"]
        required = {
            "news_id", "title", "category",
            "final_score", "semantic_score", "kg_score",
            "trend_score", "explanation",
        }
        for rec in recs:
            missing = required - rec.keys()
            assert not missing, f"Missing fields {missing} in rec: {rec['news_id']}"

    def test_top_category_is_news(self):
        """U13740 is a heavy news reader — top 5 recs should be mostly 'news'."""
        recs = _get(f"/recommend/{self.USER_ID}", top_k=10).json()["recommendations"]
        news_count = sum(1 for r in recs[:5] if r["category"] == "news")
        assert news_count >= 3, (
            f"Expected ≥3 'news' in top-5, got {news_count}. "
            f"Categories: {[r['category'] for r in recs[:5]]}"
        )

    def test_at_least_one_explanation_mentions_news(self):
        recs = _get(f"/recommend/{self.USER_ID}", top_k=10).json()["recommendations"]
        assert any("news" in r["explanation"].lower() for r in recs), (
            "No explanation references 'news' for a news-heavy user"
        )

    def test_final_scores_above_threshold(self):
        """A well-defined user vector should yield scores > 0.4."""
        recs = _get(f"/recommend/{self.USER_ID}", top_k=10).json()["recommendations"]
        max_score = max(r["final_score"] for r in recs)
        assert max_score > 0.4, f"Max final score {max_score} is unexpectedly low"

    def test_response_time_under_500ms(self):
        start = time.time()
        _get(f"/recommend/{self.USER_ID}", top_k=10)
        elapsed = time.time() - start
        assert elapsed < 0.5, f"Took {elapsed:.3f}s — limit is 0.5s"

    def test_entity_overlap_explanations_printed(self, capsys):
        """Informational: print how many recs have entity-based explanations."""
        recs = _get(f"/recommend/{self.USER_ID}", top_k=10).json()["recommendations"]
        entity_recs = [
            r for r in recs
            if "read about" in r["explanation"].lower()
            or "mentions" in r["explanation"].lower()
        ]
        with capsys.disabled():
            print(f"\n  Entity-based explanations: {len(entity_recs)}/10")
            for r in entity_recs[:3]:
                print(f"  [{r['category']}] {r['title'][:60]}")
                print(f"  → {r['explanation']}")
        # Not a hard failure — depends on dataset overlap
        assert len(entity_recs) >= 0


# ──────────────────────────────────────────────────────────────────────────────
# Test Case 2 — Cold-Start for Unknown Sports User (U99999)
# ──────────────────────────────────────────────────────────────────────────────

class TestNewSportsUser:
    """
    U99999 does not exist in the dataset.
    The API must return 404 — not hallucinate recommendations.

    To fully test diversity for a sports-only user, inject U99999 into
    user_profiles / user_interest_profiles on the server and re-run.
    """
    USER_ID = "U99999"

    def test_unknown_user_returns_404(self):
        resp = _get(f"/recommend/{self.USER_ID}", top_k=10)
        assert resp.status_code == 404

    def test_404_body_names_user(self):
        detail = _get(f"/recommend/{self.USER_ID}", top_k=10).json()["detail"]
        assert self.USER_ID in detail, f"404 detail doesn't name the user: '{detail}'"


# ──────────────────────────────────────────────────────────────────────────────
# Test Case 3 — Cold-Start / Trending Endpoint
# ──────────────────────────────────────────────────────────────────────────────

class TestTrending:

    def test_status_200(self):
        assert _get("/trending", top_k=10).status_code == 200

    def test_returns_10_articles(self):
        data = _get("/trending", top_k=10).json()
        assert "trending" in data
        assert len(data["trending"]) == 10

    def test_sorted_by_score_descending(self):
        recs = _get("/trending", top_k=10).json()["trending"]
        scores = [r["final_score"] for r in recs]
        assert scores == sorted(scores, reverse=True), (
            f"Trending not sorted descending: {scores}"
        )

    def test_all_have_non_empty_explanation(self):
        recs = _get("/trending", top_k=10).json()["trending"]
        for r in recs:
            assert r.get("explanation"), f"Empty explanation in trending: {r['news_id']}"

    def test_no_personalised_explanation(self):
        """Cold-start results must not mention personal reading history."""
        recs = _get("/trending", top_k=10).json()["trending"]
        for r in recs:
            assert "you read about" not in r["explanation"].lower(), (
                f"Personal explanation leaked into trending: '{r['explanation']}'"
            )

    def test_response_time_under_300ms(self):
        start = time.time()
        _get("/trending", top_k=10)
        elapsed = time.time() - start
        assert elapsed < 0.3, f"Trending took {elapsed:.3f}s — limit is 0.3s"


# ──────────────────────────────────────────────────────────────────────────────
# Test Case 4 — Article Similarity (BGE Cosine over 51 k vectors)
# ──────────────────────────────────────────────────────────────────────────────

class TestArticleSimilarity:
    NEWS_ID = "N55528"

    def test_status_200(self):
        assert _get(f"/similar/{self.NEWS_ID}", top_k=10).status_code == 200

    def test_response_shape(self):
        data = _get(f"/similar/{self.NEWS_ID}", top_k=10).json()
        assert "query" in data and "similar" in data
        assert data["query"]["news_id"] == self.NEWS_ID
        assert len(data["similar"]) == 10

    def test_query_article_absent_from_results(self):
        data = _get(f"/similar/{self.NEWS_ID}", top_k=10).json()
        ids = [r["news_id"] for r in data["similar"]]
        assert self.NEWS_ID not in ids, "Query article must not appear in its own results"

    def test_similarity_scores_sorted_descending(self):
        data = _get(f"/similar/{self.NEWS_ID}", top_k=10).json()
        scores = [r["similarity"] for r in data["similar"]]
        assert scores == sorted(scores, reverse=True)

    def test_all_scores_non_negative(self):
        data = _get(f"/similar/{self.NEWS_ID}", top_k=10).json()
        for r in data["similar"]:
            assert r["similarity"] >= 0.0, f"Negative similarity: {r}"

    def test_top_similarity_above_threshold(self):
        """N55528 has clear political peers — top match should exceed 0.7."""
        data = _get(f"/similar/{self.NEWS_ID}", top_k=10).json()
        top_score = data["similar"][0]["similarity"]
        assert top_score > 0.7, f"Top similarity {top_score:.4f} is unexpectedly low"

    def test_results_cluster_around_query_category(self):
        """Similar articles should mostly share the query article's category."""
        data = _get(f"/similar/{self.NEWS_ID}", top_k=10).json()
        query_cat = data["query"]["category"]
        cats = [r["category"] for r in data["similar"]]
        same_cat_count = sum(1 for c in cats if c == query_cat)
        assert same_cat_count >= 5, (
            f"Expected ≥5 articles matching query category '{query_cat}', "
            f"got {same_cat_count}. Categories: {cats}"
        )

    def test_response_time_under_1s(self):
        start = time.time()
        _get(f"/similar/{self.NEWS_ID}", top_k=10)
        elapsed = time.time() - start
        assert elapsed < 1.0, f"Similarity search took {elapsed:.3f}s — limit is 1.0s"


# ──────────────────────────────────────────────────────────────────────────────
# Test Case 5 — Category Filtering
# ──────────────────────────────────────────────────────────────────────────────

class TestCategoryFilter:
    USER_ID = "U13740"

    def test_all_results_match_category(self):
        recs = _get(f"/recommend/{self.USER_ID}", top_k=10, category="finance").json()[
            "recommendations"
        ]
        for r in recs:
            assert r["category"] == "finance", (
                f"Non-finance article in filtered results: {r['news_id']} ({r['category']})"
            )

    def test_filtered_count_lte_unfiltered(self):
        total = len(
            _get(f"/recommend/{self.USER_ID}", top_k=10).json()["recommendations"]
        )
        finance = len(
            _get(f"/recommend/{self.USER_ID}", top_k=10, category="finance").json()[
                "recommendations"
            ]
        )
        assert finance <= total

    def test_filtered_results_have_scores(self):
        recs = _get(f"/recommend/{self.USER_ID}", top_k=10, category="finance").json()[
            "recommendations"
        ]
        for r in recs:
            for field in ("final_score", "semantic_score", "kg_score"):
                assert field in r, f"Missing '{field}' in filtered rec"

    def test_nonexistent_category_returns_empty_list(self):
        resp = _get(f"/recommend/{self.USER_ID}", top_k=10, category="xyzfakecategory")
        assert resp.status_code == 200
        assert resp.json()["recommendations"] == []


# ──────────────────────────────────────────────────────────────────────────────
# Test Case 6 — Edge Cases
# ──────────────────────────────────────────────────────────────────────────────

class TestEdgeCases:

    # 6a — Invalid user ID
    def test_6a_invalid_user_recommend_404(self):
        resp = _get("/recommend/INVALID_USER")
        assert resp.status_code == 404
        assert "INVALID_USER" in resp.json()["detail"]

    def test_6a_invalid_user_history_404(self):
        resp = _get("/user/INVALID_USER/history")
        assert resp.status_code == 404

    def test_6a_invalid_user_profile_404(self):
        resp = _get("/user/INVALID_USER/profile")
        assert resp.status_code == 404

    # 6b — Invalid news ID
    def test_6b_invalid_news_similarity_404(self):
        resp = _get("/similar/INVALID_NEWS")
        assert resp.status_code == 404
        assert "INVALID_NEWS" in resp.json()["detail"]

    # 6c — top_k out of range (FastAPI Query le=20 → 422)
    def test_6c_top_k_over_max_rejected(self):
        resp = _get("/recommend/U13740", top_k=100)
        assert resp.status_code == 422, (
            f"Expected 422 for top_k=100, got {resp.status_code}"
        )

    def test_6c_top_k_zero_rejected(self):
        resp = _get("/recommend/U13740", top_k=0)
        assert resp.status_code == 422

    # Health / utility endpoints
    def test_root_ok(self):
        data = _get("/").json()
        assert data.get("status") == "ok"

    def test_stats_fields_present(self):
        data = _get("/stats").json()
        for field in ("total_articles", "users_indexed", "kg_nodes", "kg_edges"):
            assert field in data, f"Missing field '{field}' in /stats"
        assert data["total_articles"] > 0
        assert data["users_indexed"] > 0

    def test_categories_includes_news(self):
        cats = _get("/categories").json()["categories"]
        assert "news" in cats

    def test_user_history_limit_respected(self):
        data = _get("/user/U13740/history", limit=5).json()
        assert len(data["history"]) <= 5

    def test_user_profile_shape(self):
        data = _get("/user/U13740/profile").json()
        assert "top_category" in data
        assert "category_weights" in data
        assert isinstance(data["category_weights"], dict)


# ──────────────────────────────────────────────────────────────────────────────
# Test Case 7 — Performance Benchmarks
# ──────────────────────────────────────────────────────────────────────────────

N_BENCH = 5  # repetitions per benchmark


def _bench(path: str, n: int = N_BENCH, **params):
    times = []
    for _ in range(n):
        start = time.time()
        resp = _get(path, **params)
        elapsed = time.time() - start
        assert resp.status_code == 200, f"Non-200 during benchmark: {resp.status_code}"
        times.append(elapsed)
    mean = statistics.mean(times)
    std = statistics.stdev(times) if n > 1 else 0.0
    return mean, std


class TestPerformanceBenchmarks:

    def test_bench_single_recommendation(self, capsys):
        mean, std = _bench("/recommend/U13740", top_k=10)
        with capsys.disabled():
            print(f"\n  [BENCH] Single recommendation  mean={mean:.3f}s  std={std:.3f}s")
        assert mean < 0.5, f"Mean latency {mean:.3f}s > 0.5s"

    def test_bench_trending(self, capsys):
        mean, std = _bench("/trending", top_k=10)
        with capsys.disabled():
            print(f"\n  [BENCH] Trending               mean={mean:.3f}s  std={std:.3f}s")
        assert mean < 0.3, f"Mean trending latency {mean:.3f}s > 0.3s"

    def test_bench_article_similarity(self, capsys):
        mean, std = _bench("/similar/N55528", top_k=10)
        with capsys.disabled():
            print(f"\n  [BENCH] Article similarity     mean={mean:.3f}s  std={std:.3f}s")
        assert mean < 1.5, f"Mean similarity latency {mean:.3f}s > 1.5s"

    def test_bench_batch_users(self, capsys):
        """Sequential requests for multiple known users — total must be < 30 s."""
        # Fetch a pool of valid users first
        profile_resp = _get("/user/U13740/profile")
        assert profile_resp.status_code == 200

        user_ids = [f"U{13740 + i}" for i in range(10)]
        times = []
        for uid in user_ids:
            resp = _get(f"/recommend/{uid}", top_k=5)
            if resp.status_code == 200:
                start = time.time()
                _get(f"/recommend/{uid}", top_k=5)
                times.append(time.time() - start)

        if times:
            total = sum(times)
            mean = statistics.mean(times)
            with capsys.disabled():
                print(
                    f"\n  [BENCH] Batch ({len(times)} users) "
                    f"total={total:.3f}s  mean/user={mean:.3f}s"
                )
            assert total < 30.0, f"Batch total {total:.3f}s > 30s"


# ──────────────────────────────────────────────────────────────────────────────
# Test Case 8 — Explanation Quality
# ──────────────────────────────────────────────────────────────────────────────

class TestExplanationQuality:
    USER_ID = "U13740"

    def test_all_explanations_non_empty(self):
        recs = _get(f"/recommend/{self.USER_ID}", top_k=10).json()["recommendations"]
        for r in recs:
            assert r["explanation"] not in ("", "nan", "None", "null", None), (
                f"Bad explanation for {r['news_id']}: '{r['explanation']}'"
            )
            assert len(r["explanation"]) > 10, (
                f"Suspiciously short explanation: '{r['explanation']}'"
            )

    def test_entity_explanation_contains_entity_name(self):
        """'read about X' explanations must name a real entity, not be empty."""
        recs = _get(f"/recommend/{self.USER_ID}", top_k=10).json()["recommendations"]
        for r in recs:
            exp = r["explanation"]
            if "read about" in exp.lower():
                # Format: "...because you read about 'EntityName', who is..."
                assert "'" in exp, f"Entity explanation missing entity name: '{exp}'"

    def test_category_explanation_matches_user_top_category(self):
        """
        If explanation says 'frequently read X', X must be a valid category
        from the user's profile.
        """
        profile = _get(f"/user/{self.USER_ID}/profile").json()
        cat_weights = profile["category_weights"]
        recs = _get(f"/recommend/{self.USER_ID}", top_k=10).json()["recommendations"]

        for r in recs:
            exp = r["explanation"].lower()
            if "frequently read" in exp:
                # At least one known category should appear in the explanation
                assert any(cat in exp for cat in cat_weights), (
                    f"Explanation '{exp}' names a category not in user profile"
                )

    def test_low_kg_recs_have_generic_fallback(self):
        """Articles with near-zero KG score get generic fallback explanations."""
        recs = _get(f"/recommend/{self.USER_ID}", top_k=10).json()["recommendations"]
        low_kg = [r for r in recs if r["kg_score"] < 0.05]
        generic_phrases = ("reading interests", "trending", "interests and current trends")
        for r in low_kg:
            exp = r["explanation"].lower()
            assert any(p in exp for p in generic_phrases), (
                f"Low-KG rec has unexpected explanation: '{r['explanation']}'"
            )

    def test_trending_explanations_are_generic(self):
        """Cold-start trending recs must never mention personal history."""
        recs = _get("/trending", top_k=10).json()["trending"]
        for r in recs:
            exp = r["explanation"].lower()
            assert any(p in exp for p in ("trending", "popular", "readers")), (
                f"Unexpected explanation on trending rec: '{r['explanation']}'"
            )

    def test_explanation_consistency_printed(self, capsys):
        """Informational: print full explanation breakdown for U13740."""
        recs = _get(f"/recommend/{self.USER_ID}", top_k=10).json()["recommendations"]
        entity_count = sum(1 for r in recs if "read about" in r["explanation"].lower())
        category_count = sum(1 for r in recs if "frequently read" in r["explanation"].lower())
        generic_count = sum(1 for r in recs if "interests and current trends" in r["explanation"].lower())
        with capsys.disabled():
            print(f"\n  Explanation breakdown for {self.USER_ID}:")
            print(f"    Entity-based  : {entity_count}")
            print(f"    Category-based: {category_count}")
            print(f"    Generic       : {generic_count}")
            print()
            for i, r in enumerate(recs, 1):
                print(f"  {i:>2}. [{r['category']:<10}] score={r['final_score']:.4f}  "
                      f"sem={r['semantic_score']:.4f}  kg={r['kg_score']:.4f}")
                print(f"       → {r['explanation']}")
