import streamlit as st
import os
import sys
import numpy as np

# ── Path setup ─────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR  = os.path.join(BASE_DIR, "src")
sys.path.append(SRC_DIR)

from recommender import HybridNewsRecommender

# ── Page config (must be first Streamlit call) ─────────────────────────────
st.set_page_config(
    page_title = "News Recommender",
    page_icon  = "📰",
    layout     = "wide"
)

# ── Load recommender (cached — only runs once) ─────────────────────────────
@st.cache_resource
def load_recommender():
    rec = HybridNewsRecommender()
    rec.load()
    return rec

with st.spinner("Loading recommender system..."):
    recommender = load_recommender()

# ── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("📰 News Recommender")
    st.markdown("Hybrid system combining **Semantic**, **Knowledge Graph**, and **Trend** signals.")
    st.divider()

    st.subheader("📊 System Stats")
    st.metric("Total articles",    "51,282")
    st.metric("Users indexed",     "49,108")
    st.metric("KG nodes",          "71,291")
    st.metric("KG edges",          "297,352")
    st.metric("TransE entities",   "26,904")
    st.divider()

    st.subheader("🎯 Evaluation (MIND-small dev)")
    col_a, col_b = st.columns(2)
    col_a.metric("Hybrid AUC",  "0.6461")
    col_b.metric("Hybrid MRR",  "0.3677")
    col_a.metric("NDCG@5",      "0.3518")
    col_b.metric("KG AUC",      "0.6552")
    st.divider()

    st.subheader("📐 Published Baselines")
    st.markdown("""
| Model | AUC |
|---|---|
| Random | 0.5000 |
| NRMS 2019 | 0.6362 |
| **Ours** | **0.6461** ✅ |
| NAML 2019 | 0.6576 |
| UNBERT 2021 | 0.7085 |
""")
    st.divider()
    st.subheader("⚙️ Fusion Weights")
    st.markdown("""
- **Semantic** : 0.60
- **KG**       : 0.40
- **Trend**    : 0.00 *(excluded — AUC 0.599)*
""")

# ── Main title ─────────────────────────────────────────────────────────────
st.title("📰 Hybrid News Recommender")
st.caption("Built on MIND-small · BGE embeddings · TransE knowledge graph · Hybrid fusion")

# ── Tabs ───────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs([
    "👤 User Recommendations",
    "🌟 Cold Start / Trending",
    "🔍 Article Explorer"
])


# ══════════════════════════════════════════════════════════════════════════
# TAB 1 — User Recommendations
# ══════════════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("Personalised Recommendations")
    st.markdown("Enter a user ID to see personalised recommendations based on their reading history.")

    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        user_id = st.text_input("User ID", value="U13740", placeholder="e.g. U13740")
    with col2:
        top_k = st.slider("Results", min_value=5, max_value=20, value=10)
    with col3:
        all_cats = sorted(set(
            v["category"] for v in recommender.news_df_lookup.values()
        ))
        selected_cats = st.multiselect("Filter category", options=all_cats)

    get_recs = st.button("🔍 Get Recommendations", key="user_recs", type="primary")

    if get_recs:
        uid = user_id.strip()
        if not uid:
            st.warning("Please enter a user ID.")
        elif uid not in recommender.user_profiles and \
             uid not in recommender.user_interest_profiles:
            st.error(f"User **{uid}** not found. Try U13740, U53319, or any U##### ID.")
        else:
            # ── Reading history ────────────────────────────────────────────
            if uid in recommender.user_profiles:
                history = recommender.user_profiles[uid]
                with st.expander(
                    f"📚 {uid}'s reading history "
                    f"({len(history)} articles — showing last 5)"
                ):
                    for nid in history[-5:]:
                        if nid in recommender.news_df_lookup:
                            art = recommender.news_df_lookup[nid]
                            st.write(
                                f"- `{art['category'].upper()}` "
                                f"**{art['title']}**"
                            )

            # ── Top category ───────────────────────────────────────────────
            if uid in recommender.user_interest_profiles:
                top_cat = recommender.user_interest_profiles[uid].get(
                    "top_category", "unknown"
                )
                st.info(f"Top interest category: **{top_cat}**")

            # ── Recommendations ────────────────────────────────────────────
            with st.spinner("Generating recommendations..."):
                recs = recommender.recommend(uid, top_k=top_k, exclude_read=True)

            if selected_cats:
                recs = [r for r in recs if r["category"] in selected_cats]

            if not recs:
                st.info("No recommendations found for this filter combination.")
            else:
                st.success(f"Showing {len(recs)} recommendations for **{uid}**")
                st.divider()

                for i, r in enumerate(recs, 1):
                    c1, c2 = st.columns([4, 1])

                    with c1:
                        st.markdown(f"#### {i}. {r['title']}")
                        abstract = recommender.news_df_lookup.get(
                            r["news_id"], {}
                        ).get("abstract", "")
                        if abstract and abstract.lower() != "nan" and abstract.strip():
                            st.markdown(
                                f"<p style='color:#888;font-size:0.9em'>"
                                f"{abstract[:250]}{'...' if len(abstract)>250 else ''}"
                                f"</p>",
                                unsafe_allow_html=True
                            )
                        st.markdown(
                            f"🏷️ `{r['category'].upper()}`  &nbsp; "
                            f"💡 *{r['explanation']}*"
                        )

                    with c2:
                        st.metric("Final Score", f"{r['final_score']:.4f}")
                        st.write(f"🔵 Semantic : `{r['semantic_score']:.4f}`")
                        st.write(f"🟢 KG       : `{r['kg_score']:.4f}`")
                        st.write(f"🟡 Trend    : `{r['trend_score']:.4f}`")

                    st.divider()


# ══════════════════════════════════════════════════════════════════════════
# TAB 2 — Cold Start / Trending
# ══════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("Trending Articles — Cold Start")
    st.markdown(
        "For new users with no reading history, the system recommends "
        "articles ranked by **time-decay trend scores** computed from the "
        "MIND training dataset. No external news API is used."
    )

    col1, col2 = st.columns([1, 3])
    with col1:
        top_k_cs = st.slider("Number of results", 5, 20, 10, key="cs_k")

    show_trending = st.button("🌟 Show Trending Now", key="cold_start_btn", type="primary")

    if show_trending:
        with st.spinner("Fetching trending articles..."):
            recs = recommender.recommend_cold_start(top_k=top_k_cs)

        st.success(f"Top {len(recs)} trending articles")
        st.divider()

        for i, r in enumerate(recs, 1):
            c1, c2 = st.columns([4, 1])
            with c1:
                st.markdown(f"#### {i}. {r['title']}")
                abstract = recommender.news_df_lookup.get(
                    r["news_id"], {}
                ).get("abstract", "")
                if abstract and abstract.lower() != "nan" and abstract.strip():
                    st.markdown(
                        f"<p style='color:#888;font-size:0.9em'>"
                        f"{abstract[:250]}{'...' if len(abstract)>250 else ''}"
                        f"</p>",
                        unsafe_allow_html=True
                    )
                st.markdown(f"🏷️ `{r['category'].upper()}`")
                st.caption(r["explanation"])
            with c2:
                st.metric("Trend Score", f"{r['final_score']:.4f}")
            st.divider()


# ══════════════════════════════════════════════════════════════════════════
# TAB 3 — Article Explorer
# ══════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("Article Explorer — Find Similar Articles")
    st.markdown(
        "Enter a News ID to find semantically similar articles using "
        "**BGE embedding cosine similarity** across all 51,282 articles."
    )

    col1, col2 = st.columns([2, 1])
    with col1:
        news_id = st.text_input(
            "News ID", value="",
            placeholder="e.g. N12345 — find a valid ID from Tab 1 recommendations"
        )
    with col2:
        k_sim = st.slider("Similar articles", 5, 20, 10, key="sim_k")

    find_similar = st.button("🔍 Find Similar", key="similar_btn", type="primary")

    if find_similar:
        nid = news_id.strip()
        if not nid:
            st.warning("Please enter a news ID.")
        elif nid not in recommender.news_df_lookup:
            st.error(f"News ID **{nid}** not found. Copy one from the recommendations in Tab 1.")
        else:
            idx = recommender.news_id_to_idx.get(nid)
            if idx is None:
                st.error("No embedding found for this article.")
            else:
                # Show query article
                query_art = recommender.news_df_lookup[nid]
                st.info(
                    f"**Query article:** {query_art['title']} "
                    f"[`{query_art['category'].upper()}`]"
                )
                abstract = query_art.get("abstract", "")
                if abstract and abstract.lower() != "nan" and abstract.strip():
                    st.caption(abstract[:300])
                st.divider()

                # Cosine similarity against all articles
                with st.spinner("Computing similarities..."):
                    query_vec = recommender.article_embeddings[idx]
                    sims      = recommender.article_embeddings @ query_vec
                    sims      = np.maximum(sims, 0.0)
                    top_idx   = np.argsort(-sims)

                # Show top-k (skip query itself)
                st.success(f"Top {k_sim} similar articles")
                shown = 0
                for i in top_idx:
                    if shown >= k_sim:
                        break
                    candidate_id = recommender.news_id_list[i]
                    if candidate_id == nid:
                        continue

                    art = recommender.news_df_lookup.get(candidate_id, {})
                    if not art:
                        continue

                    c1, c2 = st.columns([4, 1])
                    with c1:
                        st.markdown(f"#### {shown+1}. {art['title']}")
                        abs_text = art.get("abstract", "")
                        if abs_text and abs_text.lower() != "nan" and abs_text.strip():
                            st.markdown(
                                f"<p style='color:#888;font-size:0.9em'>"
                                f"{abs_text[:250]}"
                                f"{'...' if len(abs_text)>250 else ''}"
                                f"</p>",
                                unsafe_allow_html=True
                            )
                        st.markdown(f"🏷️ `{art['category'].upper()}`")
                    with c2:
                        st.metric("Similarity", f"{float(sims[i]):.4f}")

                    st.divider()
                    shown += 1
with st.sidebar:
    st.divider()
    st.subheader("🏗️ Architecture")
    st.markdown("""
**Stage 1 — Candidate retrieval**
All 51k articles → Semantic scoring → Top 500

**Stage 2 — Full hybrid scoring**
Top 500 → KG score (4 signals) + Trend score

**Stage 3 — Fusion**
Final = 0.60 × Semantic + 0.40 × KG

**Stage 4 — Explanation**
Entity match → Category match → Fallback
""")
