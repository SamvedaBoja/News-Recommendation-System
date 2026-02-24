# 📰 Hybrid News Recommender System

A hybrid news recommendation system built on the [MIND-small dataset](https://msnews.github.io/) (Microsoft News Dataset) that combines semantic embeddings, knowledge graph reasoning, and trend scoring to generate personalised news recommendations.

---

## 🎯 Project Overview

This system addresses the challenge of personalised news recommendation by combining three complementary signals:

1. **Semantic Similarity** — Article and user embeddings via BGE (BAAI/bge-base-en-v1.5)
2. **Knowledge Graph Reasoning** — Entity matching, category weights, graph connectivity, and TransE embeddings
3. **Trend Scoring** — Time-decay and entity trend signals (computed but excluded from final fusion due to low discriminability)

### Evaluation Results (MIND-small dev set, 3000 impressions)

| Module | AUC | MRR | NDCG@5 |
|---|---|---|---|
| Semantic Only | 0.6374 | 0.3527 | 0.3378 |
| KG Only | 0.6552 | 0.3534 | 0.3389 |
| Trend Only | 0.5991 | 0.3059 | 0.2881 |
| **Hybrid (Full System)** | **0.6461** | **0.3677** | **0.3518** |

### Comparison with Published Baselines

| Model | AUC |
|---|---|
| Random | 0.5000 |
| NRMS (Wu et al., 2019) | 0.6362 |
| **Ours (Hybrid)** | **0.6461 ✅** |
| NAML (Wu et al., 2019) | 0.6576 |
| UNBERT (Zhang et al., 2021) | 0.7085 |

Our hybrid system **beats NRMS** and sits within 0.0115 of NAML — a fully neural end-to-end model — while loading in under 0.3 seconds.

---

## 🏗️ System Architecture

┌─────────────────────────────────────────────────────────┐
│ 51,282 MIND articles │
└────────────────────────┬────────────────────────────────┘
│
Stage 1: Candidate Retrieval
Semantic scoring → Top 500 candidates
│
Stage 2: Full Hybrid Scoring
┌───────────────┼───────────────┐
▼ ▼ ▼
Semantic Score KG Score Trend Score
(BGE cosine) (4 signals) (time decay +
entity trend)
│ │
└───────┬────────┘
▼
Stage 3: Weighted Fusion
Final = 0.60 × Semantic + 0.40 × KG
│
▼
Stage 4: Explanation Generation
Entity match → Category match → Fallback


### Knowledge Graph Signals (4 combined)
1. **Jaccard similarity** — exact entity overlap between user history and article
2. **Category weight** — user's reading frequency per category
3. **Graph connectivity** — shared neighbours in the KG (71,291 nodes, 297,352 edges)
4. **TransE cosine similarity** — semantic entity proximity in knowledge graph embedding space (26,904 entities, 100-dim vectors)

---

## 📁 Project Structure

news_recommender/
├── app.py # Streamlit UI
├── README.md
├── requirements.txt
│
├── src/
│ ├── recommender.py # HybridNewsRecommender class
│ ├── knowledge_graph.py # KG builder + TransE loader
│ ├── evaluation.py # Ablation study + weight tuning
│ └── preprocessing.py # Data preprocessing pipeline
│
├── data/
│ ├── train/
│ │ ├── behaviors.tsv # User click logs
│ │ ├── news.tsv # Article metadata
│ │ ├── entity_embedding.vec # Microsoft TransE vectors (100-dim)
│ │ └── relation_embedding.vec # Relation embeddings
│ ├── dev/
│ │ ├── behaviors.tsv
│ │ └── news.tsv
│ └── processed/
│ ├── news_enriched.csv # Preprocessed articles
│ ├── user_profiles.json # User reading histories
│ ├── user_interest_profiles.json
│ ├── article_embeddings.npy # BGE article vectors (51282 × 768)
│ ├── user_vectors.npy # Aggregated user vectors (49108 × 768)
│ ├── news_id_list.json
│ ├── user_vector_ids.json
│ ├── knowledge_graph.pkl # NetworkX DiGraph
│ ├── trend_data.json # Time decay + entity trend scores
│ └── evaluation_results.json
│
└── notebooks/
└── preprocessing_colab.ipynb # Colab preprocessing pipeline


---

## ⚙️ Setup & Installation

### Prerequisites
- Python 3.9+
- MIND-small dataset (download below)

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/news-recommender.git
cd news-recommender

2. Create virtual environment
python -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows

3. Install dependencies
pip install -r requirements.txt

4. Download MIND-small dataset
Download from https://msnews.github.io/ and place files under:
data/train/behaviors.tsv
data/train/news.tsv
data/train/entity_embedding.vec
data/dev/behaviors.tsv
data/dev/news.tsv

5. Run preprocessing (Google Colab recommended for embeddings)
Open notebooks/preprocessing_colab.ipynb in Google Colab and run all cells. This generates all files in data/processed/.

6. Launch the app
```bash
streamlit run app.py

🖥️ Streamlit UI
The app has three tabs:

| Tab                      | Description                                                                               |
| ------------------------ | ----------------------------------------------------------------------------------------- |
| 👤 User Recommendations  | Enter a User ID → personalised top-K recommendations with score breakdown and explanation |
| 🌟 Cold Start / Trending | New user fallback → articles ranked by time-decay trend score                             |
| 🔍 Article Explorer      | Enter a News ID → find semantically similar articles via BGE cosine similarity            |
Sample user IDs to try: U13740, U53319, U10045

📦 Requirements
streamlit>=1.32.0
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
networkx>=3.1
sentence-transformers>=2.2.0
torch>=2.0.0

🔬 Evaluation
Run the full ablation study:
```bash
python src/evaluation.py
