# RankLab MovieLens 100K benchmark

Real public ratings; a global chronological split; no sampled test negatives.

- popularity: Recall@10 0.0675, NDCG@10 0.2596, coverage 0.037, warm p95 0.54 ms.
- matrix_factorisation: Recall@10 0.0651, NDCG@10 0.2567, coverage 0.050, warm p95 0.59 ms.
- two_tower: Recall@10 0.0698, NDCG@10 0.2648, coverage 0.058, warm p95 0.64 ms.

Validation-selected model: popularity; selected MMR relevance weight: 1.0. Test results do not choose the winner.
