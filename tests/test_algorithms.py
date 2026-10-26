import numpy as np
import pytest
from ranklab.data import split, training_features, triples
from ranklab.models import bpr_loss_and_gradients, TwoTower, MatrixFactorisation
from ranklab.evaluation import (
    ranking_metrics,
    candidate_pool,
    rerank,
    paired_ndcg_interval,
)


def test_global_temporal_split_is_disjoint():
    rows = np.array([[0, i, 4, 100 + i] for i in range(100)])
    train, val, test, a, b = split(rows)
    assert (
        train[:, 3].max() < a <= val[:, 3].min()
        and val[:, 3].max() < b <= test[:, 3].min()
    )
    assert len(train) + len(val) + len(test) == 100


def test_seen_items_masked_even_when_their_score_is_highest():
    assert candidate_pool(np.array([100, 0.8, 0.7]), {0}, {0, 1, 2}).tolist() == [1, 2]


def test_recall_and_ndcg_known_examples():
    assert ranking_metrics([2, 1], [2], 2) == (1.0, 1.0)
    assert ranking_metrics([0, 1], [2], 2) == (0.0, 0.0)
    with pytest.raises(ValueError):
        ranking_metrics([1], [], 1)


def test_mmr_diversifies_without_duplicates():
    genres = np.array([[1, 0], [1, 0], [0, 1]], float)
    scores = np.array([1.0, 0.99, 0.98])
    result = rerank([0, 1, 2], scores, genres, k=2, weight=0.3)
    assert result == [0, 2]


def test_two_tower_backprop_matches_finite_difference():
    rng = np.random.default_rng(3)
    params = [
        rng.normal(size=(3, 2)) * 0.1,
        np.ones(2),
        rng.normal(size=(4, 2)) * 0.1,
        np.ones(2),
    ]
    users = rng.normal(size=(5, 3))
    pos = rng.normal(size=(5, 4))
    neg = rng.normal(size=(5, 4))
    _, grads = bpr_loss_and_gradients(params, users, pos, neg)
    for parameter, gradient in zip(params, grads):
        for index in np.ndindex(parameter.shape):
            previous = parameter[index]
            eps = 1e-6
            parameter[index] = previous + eps
            a = bpr_loss_and_gradients(params, users, pos, neg)[0]
            parameter[index] = previous - eps
            b = bpr_loss_and_gradients(params, users, pos, neg)[0]
            parameter[index] = previous
            assert np.isclose((a - b) / (2 * eps), gradient[index], atol=1e-6)


def test_training_learns_a_small_preference_problem():
    x = np.eye(2)
    positive = np.array([[0, 0, 4, 1], [1, 1, 4, 1]])
    model = TwoTower(x, x, dimensions=4).fit(
        positive, [{0}, {1}], np.array([0, 1]), epochs=80
    )
    assert model.losses[-1] < model.losses[0]
    assert (
        model.scores(0)[0] > model.scores(0)[1]
        and model.scores(1)[1] > model.scores(1)[0]
    )


def test_negative_samples_exclude_all_seen_history():
    positive = np.array([[0, 0, 4, 1]])
    _, _, negative = triples(
        positive, [{0, 1}], np.array([0, 1, 2]), np.random.default_rng(4)
    )
    assert negative.tolist() == [2]
    with pytest.raises(ValueError):
        triples(positive, [{0, 1, 2}], np.array([0, 1, 2]), np.random.default_rng(4))


def test_features_are_training_only_and_cold_users_zero():
    train = np.array([[0, 0, 4, 1], [0, 1, 2, 2]])
    _, users, items, counts, popularity, seen = training_features(train, np.eye(3), 2)
    assert counts.tolist() == [1, 0] and not users[1].any() and seen[0] == {0, 1}
    assert popularity.tolist() == [1, 0, 0]


def test_bootstrap_identity_and_pairing_contract():
    rows = [dict(user=1, ndcg=0.4), dict(user=2, ndcg=0.8)]
    assert paired_ndcg_interval(rows, rows) == dict(mean=0.0, low=0.0, high=0.0)
    with pytest.raises(ValueError):
        paired_ndcg_interval(rows, rows[:1])


def test_matrix_factorisation_outputs_finite_scores():
    positive = np.array([[0, 0, 4, 1], [1, 1, 4, 1]])
    model = MatrixFactorisation(2, 3, dimensions=4).fit(
        positive, [{0}, {1}], np.array([0, 1, 2]), epochs=3
    )
    assert np.isfinite(model.scores(0)).all() and model.scores(0)[2] == 0


def test_packaged_data_manifest_is_available():
    import importlib.resources, json

    source = json.loads(
        importlib.resources.files("ranklab").joinpath("source.json").read_text()
    )
    assert len(source["sha256"]) == 64
