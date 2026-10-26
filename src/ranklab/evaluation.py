import time
import numpy as np


def ranking_metrics(recommendations, relevant, k=10):
    relevant = set(relevant)
    if not relevant:
        raise ValueError("Need at least one relevant held-out item")
    hits = np.array([int(i in relevant) for i in recommendations[:k]])
    recall = float(hits.sum() / len(relevant))
    ideal = np.sum(1 / np.log2(np.arange(min(k, len(relevant))) + 2))
    ndcg = float(np.sum(hits / np.log2(np.arange(len(hits)) + 2)) / ideal)
    return recall, ndcg


def candidate_pool(scores, seen, catalog, size=100):
    allowed = np.array([i for i in catalog if i not in seen], int)
    # Deterministic item-index tie break keeps popularity results reproducible.
    return allowed[np.lexsort((allowed, -np.asarray(scores)[allowed]))[:size]]


def rerank(pool, scores, genres, k=10, weight=0.8):
    if not 0 <= weight <= 1:
        raise ValueError("MMR weight must be in [0,1]")
    pool = np.asarray(pool, int)
    chosen = []
    if not len(pool):
        return chosen
    values = np.asarray(scores)[pool]
    span = max(float(values.max() - values.min()), 1e-12)
    relevance = (values - values.min()) / span
    normal = genres[pool] / np.maximum(
        np.linalg.norm(genres[pool], axis=1, keepdims=True), 1e-12
    )
    similarities = normal @ normal.T
    penalty = np.zeros(len(pool))
    available = np.ones(len(pool), bool)
    while available.any() and len(chosen) < k:
        utility = weight * relevance - (1 - weight) * penalty
        utility[~available] = -np.inf
        best = int(np.lexsort((pool, -utility))[0])
        chosen.append(int(pool[best]))
        available[best] = False
        penalty = np.maximum(penalty, similarities[:, best])
    return chosen


def evaluate(
    scores_fn,
    window,
    seen,
    catalog,
    genres,
    train_counts,
    positive_counts,
    k=10,
    weight=1.0,
):
    relevant = {}
    for user, item, rating, _ in window:
        # Known items are masked from retrieval, so they are not eligible labels.
        if rating >= 4 and item in catalog and item not in seen[user]:
            relevant.setdefault(int(user), set()).add(int(item))
    if not relevant:
        raise ValueError("No eligible positive labels in this time window")
    rows = []
    latencies = []
    exposed = set()
    raw_positive_rows = int((window[:, 2] >= 4).sum())
    for user, truth in sorted(relevant.items()):
        # Warm up scoring outside latency measurements.
        scores_fn(user)
        start = time.perf_counter_ns()
        scores = scores_fn(user)
        pool = candidate_pool(scores, seen[user], catalog, 100)
        recommendations = (
            rerank(pool, scores, genres, k, weight) if weight < 1 else pool[:k].tolist()
        )
        latencies.append((time.perf_counter_ns() - start) / 1e6)
        recall, ndcg = ranking_metrics(recommendations, truth, k)
        normal = genres[recommendations] / np.maximum(
            np.linalg.norm(genres[recommendations], axis=1, keepdims=True), 1e-12
        )
        pairwise = normal @ normal.T
        diversity = (
            float((1 - pairwise)[np.triu_indices(len(recommendations), 1)].mean())
            if len(recommendations) > 1
            else 0.0
        )
        cold = {item for item in truth if positive_counts[item] == 0}
        rows.append(
            dict(
                user=user,
                recall=recall,
                ndcg=ndcg,
                diversity=diversity,
                candidate_recall=len(set(pool) & truth) / len(truth),
                cold_user=train_counts[user] == 0,
                cold_item_relevant=len(cold),
                cold_item_hits=len(set(recommendations) & cold),
            )
        )
        exposed.update(recommendations)

    def metrics(group):
        return dict(
            users=len(group),
            recall_at_10=float(np.mean([r["recall"] for r in group]))
            if group
            else None,
            ndcg_at_10=float(np.mean([r["ndcg"] for r in group])) if group else None,
        )

    result = metrics(rows)
    result.update(
        candidate_recall_at_100=float(np.mean([r["candidate_recall"] for r in rows])),
        mean_genre_diversity=float(np.mean([r["diversity"] for r in rows])),
        catalogue_coverage=len(exposed) / len(catalog),
        p50_ms=float(np.percentile(latencies, 50)),
        p95_ms=float(np.percentile(latencies, 95)),
        warm_users=metrics([r for r in rows if not r["cold_user"]]),
        cold_users=metrics([r for r in rows if r["cold_user"]]),
        cold_item_relevant=sum(r["cold_item_relevant"] for r in rows),
        cold_item_hits=sum(r["cold_item_hits"] for r in rows),
        raw_positive_rows=raw_positive_rows,
        eligible_unique_positive_pairs=sum(len(s) for s in relevant.values()),
    )
    return result, rows


def paired_ndcg_interval(a, b, seed=41):
    left = {r["user"]: r["ndcg"] for r in a}
    right = {r["user"]: r["ndcg"] for r in b}
    if left.keys() != right.keys():
        raise ValueError("Paired user sets must match")
    delta = np.array([right[u] - left[u] for u in sorted(left)])
    rng = np.random.default_rng(seed)
    draws = [rng.choice(delta, len(delta), replace=True).mean() for _ in range(1000)]
    return dict(
        mean=float(delta.mean()),
        low=float(np.quantile(draws, 0.025)),
        high=float(np.quantile(draws, 0.975)),
    )
