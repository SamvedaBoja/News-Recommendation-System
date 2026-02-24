import pandas as pd
import numpy as np
import json
import os
import sys

# Add src to path so we can import recommender
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from recommender import HybridNewsRecommender, safe_parse, W_KG, W_SEMANTIC, W_TREND

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED = os.path.join(BASE_DIR, "data", "processed")

# Check for validation data
VALID_DIR = None
for candidate in ["data/valid", "data/dev",
                  "data/MINDsmall_dev", "data/MINDsmall_valid"]:
    full = os.path.join(BASE_DIR, candidate)
    if os.path.exists(full):
        VALID_DIR = full
        break

if VALID_DIR is None:
    print("⚠️  No validation folder found — using train behaviors")
    VALID_DIR = os.path.join(BASE_DIR, "data", "train")

BEHAVIOR_COLS = ["impression_id", "user_id", "time", "history", "impressions"]


# ── Metric Functions ───────────────────────────────────────────────────────
def compute_auc(labels, scores):
    if len(set(labels)) < 2:
        return None
    paired  = sorted(zip(scores, labels), reverse=True)
    n_pos   = sum(labels)
    n_neg   = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return None
    tp, fp, auc_val, prev_fp = 0, 0, 0.0, 0
    for score, label in paired:
        if label == 1:
            tp += 1
        else:
            fp += 1
            auc_val += tp * (fp - prev_fp)
            prev_fp  = fp
    return round(auc_val / (n_pos * n_neg), 4)


def compute_mrr(labels, scores):
    paired = sorted(zip(scores, labels), reverse=True)
    for rank, (_, label) in enumerate(paired, 1):
        if label == 1:
            return round(1.0 / rank, 4)
    return 0.0


def compute_ndcg(labels, scores, k=5):
    paired = sorted(zip(scores, labels), reverse=True)[:k]
    dcg    = sum(label / np.log2(rank + 1)
                 for rank, (_, label) in enumerate(paired, 1))
    ideal  = sorted(labels, reverse=True)[:k]
    idcg   = sum(label / np.log2(rank + 1)
                 for rank, label in enumerate(ideal, 1))
    return round(dcg / idcg, 4) if idcg > 0 else 0.0


# ── Helper: Build Fresh User Vector ───────────────────────────────────────
def _build_fresh_user_vector(history_list, recommender, decay=0.9):
    vectors, weights = [], []
    n = len(history_list)
    for i, nid in enumerate(history_list):
        if nid in recommender.news_id_to_idx:
            idx = recommender.news_id_to_idx[nid]
            vectors.append(recommender.article_embeddings[idx])
            weights.append(decay ** (n - i - 1))
    if not vectors:
        return None
    vectors = np.array(vectors)
    weights = np.array(weights)
    weights = weights / weights.sum()
    vec     = np.average(vectors, axis=0, weights=weights)
    norm    = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec


# ── Helper: Build Fresh Interest Profile ──────────────────────────────────
def _build_fresh_profile(user_history, recommender):
    history_entities, history_cats = [], []
    for nid in user_history:
        if nid in recommender.news_df_lookup:
            art = recommender.news_df_lookup[nid]
            history_cats.append(art.get("category", ""))
            history_entities.extend(art.get("entity_ids", []))
    cat_series  = pd.Series(history_cats).value_counts()
    cat_weights = (cat_series / cat_series.sum()).to_dict() \
                  if len(cat_series) > 0 else {}
    return cat_weights, set(history_entities)


# ── Helper: Build User TransE Mean Vector ─────────────────────────────────
def _build_user_transe_vec(entity_set, transe_embeddings):
    """Compute L2-normalised mean TransE vector for a set of entities."""
    vecs = [transe_embeddings[e] for e in entity_set
            if e in transe_embeddings]
    if not vecs:
        return None
    mean = np.mean(vecs, axis=0)
    norm = np.linalg.norm(mean)
    return mean / norm if norm > 0 else mean


# ── Helper: Score Candidates ───────────────────────────────────────────────
def _score_candidates(recommender, user_id, news_ids,
                      fresh_user_vec, user_history, mode,
                      w_kg=None, w_sem=None, w_trd=None):
    wk = w_kg  if w_kg  is not None else W_KG
    ws = w_sem if w_sem is not None else W_SEMANTIC
    wt = w_trd if w_trd is not None else W_TREND

    cat_weights, entity_set = _build_fresh_profile(user_history, recommender)

    user_transe_mean = _build_user_transe_vec(
        entity_set, recommender.transe_embeddings
    ) if hasattr(recommender, "transe_embeddings") and \
         recommender.transe_embeddings else None

    scores = []
    for nid in news_ids:
        s, k, t = 0.0, 0.0, 0.0

        # Semantic
        if fresh_user_vec is not None and nid in recommender.news_id_to_idx:
            idx = recommender.news_id_to_idx[nid]
            s   = float(max(np.dot(
                fresh_user_vec, recommender.article_embeddings[idx]), 0.0))

        if nid in recommender.news_df_lookup:
            art           = recommender.news_df_lookup[nid]
            art_entities  = set(art.get("entity_ids", []))
            art_cat       = art.get("category", "")
            art_ents_list = art.get("entity_ids", [])

            # KG Signal 1: Jaccard
            union   = entity_set | art_entities
            jaccard = len(entity_set & art_entities) / len(union) \
                      if len(union) > 0 else 0.0

            # KG Signal 2: Category weight
            cat_b = cat_weights.get(art_cat, 0.0)

            # KG Signal 3: Graph connectivity
            graph_b = 0.0
            if (recommender.kg_graph.has_node(user_id) and
                    recommender.kg_graph.has_node(nid)):
                if recommender.kg_graph.has_edge(user_id, nid):
                    graph_b = 1.0
                else:
                    u_nb    = set(recommender.kg_graph.successors(user_id))
                    a_nb    = set(recommender.kg_graph.predecessors(nid))
                    graph_b = min(len(u_nb & a_nb) * 0.1, 0.5)

            # KG Signal 4: TransE
            transe_sim = 0.0
            if user_transe_mean is not None:
                art_transe_mean = _build_user_transe_vec(
                    art_entities, recommender.transe_embeddings
                )
                if art_transe_mean is not None:
                    transe_sim = float(
                        np.dot(user_transe_mean, art_transe_mean)
                    )
                    transe_sim = max(transe_sim, 0.0)

            k = (0.30 * jaccard) + \
                (0.25 * cat_b)   + \
                (0.20 * graph_b) + \
                (0.25 * transe_sim)

            # Trend
            cat_w     = cat_weights.get(art_cat, 0.05)
            time_d    = recommender.time_decay_scores.get(nid, 0.1)
            ent_boost = float(np.mean([
                recommender.entity_trend_scores.get(eid, 0.0)
                for eid in art_ents_list
            ])) if art_ents_list else 0.0
            t = (0.6 * cat_w * time_d) + (0.4 * ent_boost)

        if mode == "hybrid":
            scores.append((wk * k) + (ws * s) + (wt * t))
        elif mode == "semantic":
            scores.append(s)
        elif mode == "kg":
            scores.append(k)
        elif mode == "trend":
            scores.append(t)

    return scores



# ── Main Evaluation Loop ───────────────────────────────────────────────────
def evaluate(recommender, behaviors, max_impressions=3000):
    behaviors = behaviors[
        behaviors["imp_list"].apply(
            lambda x: any(l == 1 for _, l in x) and
                      any(l == 0 for _, l in x)
        )
    ].reset_index(drop=True)

    if len(behaviors) > max_impressions:
        behaviors = behaviors.sample(
            n=max_impressions, random_state=42
        ).reset_index(drop=True)

    print(f"Evaluating on {len(behaviors)} impressions...\n")

    modes   = ["hybrid", "semantic", "kg", "trend"]
    results = {m: {"auc": [], "mrr": [], "ndcg5": []} for m in modes}

    for idx, (_, row) in enumerate(behaviors.iterrows()):
        if idx % 500 == 0:
            print(f"  Progress: {idx}/{len(behaviors)}")

        user_id      = row["user_id"]
        imp_list     = row["imp_list"]
        history_str  = str(row.get("history", ""))
        user_history = history_str.strip().split() \
                       if history_str.strip() else []
        fresh_vec    = _build_fresh_user_vector(user_history, recommender)
        news_ids     = [nid for nid, _ in imp_list]
        labels       = [lbl for _, lbl in imp_list]

        for mode in modes:
            scores = _score_candidates(
                recommender, user_id, news_ids,
                fresh_vec, user_history, mode
            )
            auc  = compute_auc(labels, scores)
            mrr  = compute_mrr(labels, scores)
            ndcg = compute_ndcg(labels, scores, k=5)
            if auc is not None:
                results[mode]["auc"].append(auc)
            results[mode]["mrr"].append(mrr)
            results[mode]["ndcg5"].append(ndcg)

    return results


# ── Weight Tuning ──────────────────────────────────────────────────────────
def tune_weights(recommender, behaviors_sample):
    print("\n── Weight Tuning (500 impressions) ────────────────────────")
    print(f"  {'W_KG':>6} {'W_SEM':>6} {'W_TRD':>6} {'AUC':>8} {'MRR':>8}")
    print("  " + "-" * 45)

    weight_combos = [
        (0.45, 0.55, 0.00),   # KG + Semantic only
        (0.50, 0.50, 0.00),   # equal split
        (0.40, 0.60, 0.00),   # semantic heavy
        (0.55, 0.45, 0.00),   # KG heavy
        (0.60, 0.40, 0.00),   # KG dominant
        (0.35, 0.55, 0.10),   # tiny trend
    ]


    best_auc     = 0.0
    best_weights = weight_combos[0]

    sample = behaviors_sample[
        behaviors_sample["imp_list"].apply(
            lambda x: any(l == 1 for _, l in x) and
                      any(l == 0 for _, l in x)
        )
    ].head(500)

    for w_kg, w_sem, w_trd in weight_combos:
        aucs, mrrs = [], []

        for _, row in sample.iterrows():
            user_id      = row["user_id"]
            imp_list     = row["imp_list"]
            history_str  = str(row.get("history", ""))
            user_history = history_str.strip().split() \
                           if history_str.strip() else []
            fresh_vec    = _build_fresh_user_vector(user_history, recommender)
            news_ids     = [nid for nid, _ in imp_list]
            labels       = [lbl for _, lbl in imp_list]

            scores = _score_candidates(
                recommender, user_id, news_ids,
                fresh_vec, user_history, "hybrid",
                w_kg=w_kg, w_sem=w_sem, w_trd=w_trd
            )
            auc = compute_auc(labels, scores)
            mrr = compute_mrr(labels, scores)
            if auc is not None:
                aucs.append(auc)
            mrrs.append(mrr)

        avg_auc = np.mean(aucs) if aucs else 0.0
        avg_mrr = np.mean(mrrs) if mrrs else 0.0
        marker  = " ← best so far" if avg_auc > best_auc else ""

        print(f"  {w_kg:>6} {w_sem:>6} {w_trd:>6} "
              f"{avg_auc:>8.4f} {avg_mrr:>8.4f}{marker}")

        if avg_auc > best_auc:
            best_auc     = avg_auc
            best_weights = (w_kg, w_sem, w_trd)

    print("\n" + "="*55)
    print(f"  ✅ Best weights → W_KG={best_weights[0]}, "
          f"W_SEM={best_weights[1]}, W_TRD={best_weights[2]}")
    print(f"  ✅ Best AUC     → {best_auc:.4f}")
    print("="*55)
    print("\n  👉 Update these 3 lines in src/recommender.py:")
    print(f"     W_KG       = {best_weights[0]}")
    print(f"     W_SEMANTIC = {best_weights[1]}")
    print(f"     W_TREND    = {best_weights[2]}")

    return best_weights


# ── Print Results Table ────────────────────────────────────────────────────
def print_results(results):
    print("\n" + "="*65)
    print("ABLATION STUDY — Evaluation Results")
    print("="*65)
    print(f"  {'Module':<20} {'AUC':>10} {'MRR':>10} {'NDCG@5':>10}")
    print("-"*65)

    mode_labels = {
        "semantic": "Semantic Only",
        "kg"      : "KG Only",
        "trend"   : "Trend Only",
        "hybrid"  : "Hybrid (Full System)"
    }

    best_mode = max(
        ["semantic", "kg", "trend", "hybrid"],
        key=lambda m: np.mean(results[m]["auc"]) if results[m]["auc"] else 0
    )

    for mode in ["semantic", "kg", "trend", "hybrid"]:
        r    = results[mode]
        auc  = np.mean(r["auc"])   if r["auc"]   else 0.0
        mrr  = np.mean(r["mrr"])   if r["mrr"]   else 0.0
        ndcg = np.mean(r["ndcg5"]) if r["ndcg5"] else 0.0
        marker = " ← Best" if mode == best_mode else ""
        print(f"  {mode_labels[mode]:<20} {auc:>10.4f} "
              f"{mrr:>10.4f} {ndcg:>10.4f}{marker}")

    print("="*65)
    print("\nMetric Definitions:")
    print("  AUC    — separates clicked vs non-clicked (0.5=random, 1.0=perfect)")
    print("  MRR    — rank of first clicked article (1.0=always ranked first)")
    print("  NDCG@5 — quality of top-5 ranking (higher=better)")

    print("\nContext vs MIND-small published baselines:")
    print("  Random baseline : AUC=0.5000")
    print("  NRMS (2019)     : AUC=0.6362")
    print("  NAML (2019)     : AUC=0.6576")
    print("  UNBERT (2021)   : AUC=0.7085")

    # Save results
    summary = {}
    for mode in results:
        r = results[mode]
        summary[mode] = {
            "AUC"   : round(float(np.mean(r["auc"])),   4) if r["auc"]   else 0.0,
            "MRR"   : round(float(np.mean(r["mrr"])),   4) if r["mrr"]   else 0.0,
            "NDCG@5": round(float(np.mean(r["ndcg5"])), 4) if r["ndcg5"] else 0.0
        }
    out_path = os.path.join(PROCESSED, "evaluation_results.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n✅ Results saved to data/processed/evaluation_results.json")


# ── Entry Point ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import time

    recommender = HybridNewsRecommender()
    recommender.load()

    print(f"Loading behaviors from: {VALID_DIR}")
    behaviors_df = pd.read_csv(
        os.path.join(VALID_DIR, "behaviors.tsv"),
        sep="\t", header=None, names=BEHAVIOR_COLS
    )
    print(f"Total impressions available: {len(behaviors_df)}")

    def parse_imp(s):
        if pd.isna(s): return []
        result = []
        for item in str(s).strip().split():
            parts = item.split("-")
            if len(parts) == 2:
                try: result.append((parts[0], int(parts[1])))
                except: pass
        return result

    behaviors_df["imp_list"] = behaviors_df["impressions"].apply(parse_imp)

    # ── Step 1: Weight tuning ──────────────────────────────────────────────
    print("\n" + "="*55)
    print("STEP 1: Weight Tuning")
    print("="*55)
    best_weights = tune_weights(recommender, behaviors_df)

    # ── Step 2: Full evaluation ────────────────────────────────────────────
    print("\n" + "="*55)
    print("STEP 2: Full Evaluation (3000 impressions)")
    print("="*55)
    t0      = time.time()
    results = evaluate(recommender, behaviors_df, max_impressions=3000)
    elapsed = time.time() - t0

    print_results(results)
    print(f"\n⏱  Total evaluation time: {elapsed:.1f}s")
