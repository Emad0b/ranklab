# RankLab

A reproducible recommendation-system benchmark with popularity, pairwise matrix factorisation, a nonlinear two-tower model and diversity-aware re-ranking. Real MovieLens 100K data; no website or paid API required.

## Implemented

- Global chronological train/validation/test boundaries, not random rating splits.
- Positive preference defined as a rating of at least four stars.
- BPR matrix factorisation with learned item bias.
- Two nonlinear towers with explicit NumPy backpropagation and Adam; user taste profiles and item genre/popularity features.
- Top-100 candidate retrieval, top-10 ranking and maximal marginal relevance (MMR) re-ranking.
- Recall@10, NDCG@10, candidate recall, genre diversity, catalogue coverage and warm p50/p95 latency.
- Cold-user fallback, cold-item outcomes and paired user-bootstrap intervals.
- Validation-only model/MMR selection; all final-test comparisons remain visible.

## Run

Python 3.12 recommended. From the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
$env:OPENBLAS_NUM_THREADS='1'
python -m ranklab.fetch_data
python -m ranklab.cli
python -m pytest -q
```

On macOS/Linux use `source .venv/bin/activate` and `export OPENBLAS_NUM_THREADS=1`. The downloader verifies a pinned SHA-256 and retains TLS verification. It can verify an already-downloaded archive at `data/raw/ml-100k.zip` if network access is unavailable. `--epochs` changes the training budget; the recorded benchmark uses 12. Changing it creates a new experiment and must not be tuned repeatedly against the final test.

## Evaluation contract

Timestamp quantiles define global boundaries near 70% and 85% of the rating history; ties remain on one side of a boundary. Models, feature profiles, popularity counts and seen-item masks use only the training window. They are frozen for both later windows. No validation interactions are folded into test-time history.

The catalogue is frozen using movie release dates before the training cutoff. Missing release dates are excluded. This assumes that published movie metadata would be available; the downloaded metadata is a retrospective snapshot, not an actual historical catalogue service. No user demographic fields are used.

Every rated training item is masked, including low-rated ones. Eligible relevant labels are future ratings >=4 for eligible, previously unseen items. Users without an eligible positive are excluded from ranking metrics; counts and raw/eligible label totals are reported. Test negatives are not sampled: rank against the entire eligible unseen catalogue. Training uses one seeded unobserved negative per positive, excluding all rated training items. Unobserved items are not known dislikes.

The content-based two-tower model is distinct from ID-based matrix factorisation: it can score eligible unseen items using genres. Users without positive training history use the same popularity fallback across models. Cold-item metrics define cold as no positive training interaction. Unknown-user fallback is evaluated among users present in the later log; unseen future identities are not assumed known preferences.

Candidate retrieval here uses exact dot-product scoring and sorting over roughly 1,700 items. It is not a distributed approximate-nearest-neighbour service. MMR trades normalised relevance against maximum genre cosine similarity to previously selected items. Weights 0.6, 0.8 and 1.0 are compared; validation NDCG chooses the default. The selected weight can legitimately be 1.0, meaning no diversity penalty.

## Results and source

- [Measured comparison](reports/summary.md)
- [Complete metrics and loss histories](reports/results.json)
- [Dependency and source fingerprints](reports/provenance.json)
- [Algorithms](src/ranklab/models.py)
- [Evaluation](src/ranklab/evaluation.py)

Latency includes warm local scoring, full sorting and optional MMR. It excludes training, network transport and data loading. These measurements are hardware-specific, not a service SLA. The paired bootstrap resamples users conditional on the trained models and one temporal split; it does not include training-seed or time-window uncertainty.

## Tests

Tests verify the global time split, seen-item masking, known ranking metrics, MMR diversity, finite-difference gradients, learning on a small preference problem, legal negative samples, training-only profiles, paired intervals and finite MF scores. CI uses small fixtures and does not download or redistribute MovieLens. The workflow is included; a hosted CI result is not claimed.

## Limits and next work

MovieLens is an old, selected ratings sample, not an exposure or click log. Offline ranking quality cannot establish causal engagement or online lift. Fixed training-history evaluation differs from a continuously updated service. Useful next steps are multiple rolling cutoffs, training-seed repetitions, ANN recall/latency comparisons, item-text features and a real online experiment.

## Data rights and attribution

GroupLens Research, University of Minnesota, MovieLens 100K: https://grouplens.org/datasets/movielens/100k/ . Citation: F. Maxwell Harper and Joseph A. Konstan (2015), *The MovieLens Datasets: History and Context*, DOI https://doi.org/10.1145/2827872 .

MovieLens has research-use conditions, requires attribution, prohibits raw redistribution without permission and requires separate permission for commercial/revenue-bearing use. Review the archive's README before reuse. Raw data, user/item-level exports and learned weights are excluded from Git. This repository contains code and aggregate research results. No endorsement is implied. MIT applies to our code, not the dataset.

## Verified result

11 tests passed. The benchmark used 100,000 real MovieLens ratings. Popularity won validation; the neural model was not promoted based on its final-test score. [Benchmark CSV](reports/benchmark.csv) contains the comparisons.

[Decision memo](docs/decision-memo.md) explains the evidence and limits.

For the recorded Python 3.12 versions, run `pip install -r requirements-lock.txt` then `pip install -e . --no-deps`.
