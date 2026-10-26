import csv
import argparse, json, hashlib, csv
from pathlib import Path
import numpy as np
from .data import load, split, training_features
from .models import MatrixFactorisation, TwoTower
from .evaluation import evaluate, paired_ndcg_interval
from .provenance import capture

ROOT = Path.cwd()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "reports")
    parser.add_argument("--epochs", type=int, default=12)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    source = json.loads(
        Path(__file__).with_name("source.json").read_text(encoding="utf-8")
    )
    ratings, genres, released = load(
        ROOT / "data/raw" / source["filename"], source["sha256"]
    )
    train, validation, test, a, b = split(ratings)
    positives, user, item, counts, popularity, seen = training_features(
        train, genres, int(ratings[:, 0].max() + 1)
    )
    # Freeze catalogue at the training cutoff; release date is not a rating label.
    catalog = set(np.flatnonzero(released < a).tolist())
    catalog_array = np.array(sorted(catalog))
    positives = positives[np.isin(positives[:, 1], catalog_array)]
    print(
        "Training on",
        len(train),
        "ratings;",
        len(positives),
        "positive examples;",
        len(catalog),
        "eligible items",
        flush=True,
    )
    mf = MatrixFactorisation(len(user), len(item)).fit(
        positives, seen, catalog_array, epochs=args.epochs
    )
    tower = TwoTower(user, item).fit(positives, seen, catalog_array, epochs=args.epochs)

    def wrap(model):
        return lambda u: model.scores(u) if counts[u] > 0 else popularity.copy()

    scorers = dict(
        popularity=lambda u: popularity.copy(),
        matrix_factorisation=wrap(mf),
        two_tower=wrap(tower),
    )
    results = {}
    test_rows = {}
    selected = None
    for name, scorer in scorers.items():
        val, _ = evaluate(scorer, validation, seen, catalog, genres, counts, popularity)
        results[name] = {"validation": val}
    selected = max(results, key=lambda n: results[n]["validation"]["ndcg_at_10"])
    # MMR setting is selected on validation NDCG; all trade-offs remain visible.
    tradeoffs = {}
    for weight in [0.6, 0.8, 1.0]:
        val, _ = evaluate(
            scorers[selected],
            validation,
            seen,
            catalog,
            genres,
            counts,
            popularity,
            weight=weight,
        )
        tradeoffs[str(weight)] = {"validation": val}
    chosen_weight = float(
        max(tradeoffs, key=lambda w: tradeoffs[w]["validation"]["ndcg_at_10"])
    )
    for name, scorer in scorers.items():
        result, rows = evaluate(scorer, test, seen, catalog, genres, counts, popularity)
        results[name]["test"] = result
        test_rows[name] = rows
    for name in results:
        results[name]["test_minus_popularity_ndcg_95ci"] = paired_ndcg_interval(
            test_rows["popularity"], test_rows[name]
        )
    for weight in tradeoffs:
        tradeoffs[weight]["test"] = evaluate(
            scorers[selected],
            test,
            seen,
            catalog,
            genres,
            counts,
            popularity,
            weight=float(weight),
        )[0]
    report = dict(
        source=source,
        protocol=dict(
            train_cutoff=a,
            test_cutoff=b,
            train_rows=len(train),
            validation_rows=len(validation),
            test_rows=len(test),
            epochs=args.epochs,
            seed=17,
            positive_rating_at_least=4,
            catalogue_items=len(catalog),
            evaluation="Frozen training history; full eligible catalogue, mask every previously rated training item; no sampled test negatives.",
        ),
        models=results,
        validation_selected=selected,
        mmr_validation_selected_weight=chosen_weight,
        mmr=tradeoffs,
        selected_minus_popularity_ndcg_bootstrap=paired_ndcg_interval(
            test_rows["popularity"], test_rows[selected]
        ),
        losses=dict(matrix_factorisation=mf.losses, two_tower=tower.losses),
        limitations="Ratings are not impression logs. Unobserved items are not known negatives. Offline relevance is not causal engagement or online lift. Latency is warm local NumPy scoring plus full sorting/MMR, not a service SLA.",
    )
    (args.output / "results.json").write_text(
        json.dumps(report, indent=2, allow_nan=False)
    )
    with (args.output / "benchmark.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "model",
                "window",
                "recall_at_10",
                "ndcg_at_10",
                "catalogue_coverage",
                "p95_ms",
            ]
        )
        for name, value in results.items():
            for window in ["validation", "test"]:
                r = value[window]
                writer.writerow(
                    [
                        name,
                        window,
                        r["recall_at_10"],
                        r["ndcg_at_10"],
                        r["catalogue_coverage"],
                        r["p95_ms"],
                    ]
                )
    text = "# RankLab MovieLens 100K benchmark\n\nReal public ratings; a global chronological split; no sampled test negatives.\n\n"
    for name, result in results.items():
        r = result["test"]
        text += f"- {name}: Recall@10 {r['recall_at_10']:.4f}, NDCG@10 {r['ndcg_at_10']:.4f}, coverage {r['catalogue_coverage']:.3f}, warm p95 {r['p95_ms']:.2f} ms.\n"
    text += f"\nValidation-selected model: {selected}; selected MMR relevance weight: {chosen_weight}. Test results do not choose the winner.\n"
    (args.output / "summary.md").write_text(text, encoding="utf-8")
    capture(args.output, source)
    print(text)
    artifacts = ROOT / "artifacts"
    artifacts.mkdir(exist_ok=True)
    np.savez(
        artifacts / "two_tower.npz",
        **{f"p{i}": v for i, v in enumerate(tower.parameters)},
    )
    np.savez(
        artifacts / "matrix_factorisation.npz",
        users=mf.users,
        items=mf.items,
        bias=mf.bias,
    )


if __name__ == "__main__":
    main()
