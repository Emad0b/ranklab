# RankLab decision memo

The validation-selected model is **popularity**. The two-tower model scores slightly higher on the final test, but this does not change the selection. Its paired user-bootstrap NDCG difference from popularity is 0.0052, with 95% interval [-0.0008, 0.0122]. This interval conditions on trained models and one temporal split; it excludes training-seed and time-window uncertainty.

Do not retrospectively choose the neural model from its test score. A new prespecified rolling evaluation is required for a new promotion decision. The validation-selected MMR relevance weight is 1.0; the report preserves the diversity trade-offs even when a penalty reduces relevance.

Warm NumPy latency over a small catalogue is not evidence of distributed serving readiness. No online engagement improvement is claimed from these offline ratings.
