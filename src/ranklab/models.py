import numpy as np
from .data import triples


def bpr_loss_and_gradients(parameters, users, positive, negative, reg=1e-4):
    """A real nonlinear two-tower model, implemented with explicit NumPy backprop."""
    wu, bu, wi, bi = parameters
    zu = users @ wu + bu
    zp = positive @ wi + bi
    zn = negative @ wi + bi
    u = np.maximum(zu, 0)
    p = np.maximum(zp, 0)
    n = np.maximum(zn, 0)
    scale = np.sqrt(u.shape[1])
    difference = np.sum(u * (p - n), axis=1) / scale
    g = 1 / (1 + np.exp(np.clip(difference, -40, 40)))
    du = g[:, None] * (n - p) / scale * (zu > 0)
    dp = -g[:, None] * u / scale * (zp > 0)
    dn = g[:, None] * u / scale * (zn > 0)
    count = len(users)
    gradients = [
        users.T @ du / count + reg * wu,
        du.mean(axis=0),
        (positive.T @ dp + negative.T @ dn) / count + reg * wi,
        (dp + dn).mean(axis=0),
    ]
    loss = np.logaddexp(0, -difference).mean() + reg / 2 * (
        np.square(wu).sum() + np.square(wi).sum()
    )
    return float(loss), gradients


class TwoTower:
    def __init__(self, user_features, item_features, seed=17, dimensions=24):
        self.user_features = user_features
        self.item_features = item_features
        rng = np.random.default_rng(seed)
        self.parameters = [
            rng.normal(0, 0.15, (user_features.shape[1], dimensions)),
            np.full(dimensions, 0.1),
            rng.normal(0, 0.15, (item_features.shape[1], dimensions)),
            np.full(dimensions, 0.1),
        ]
        self.losses = []

    def fit(self, positives, seen, catalog, epochs=12, seed=18):
        rng = np.random.default_rng(seed)
        u, p, n = triples(positives, seen, catalog, rng)
        first = [np.zeros_like(x) for x in self.parameters]
        second = [np.zeros_like(x) for x in self.parameters]
        step = 0
        for _ in range(epochs):
            losses = []
            for batch in np.array_split(rng.permutation(len(u)), max(1, len(u) // 256)):
                loss, gradients = bpr_loss_and_gradients(
                    self.parameters,
                    self.user_features[u[batch]],
                    self.item_features[p[batch]],
                    self.item_features[n[batch]],
                )
                step += 1
                losses.append(loss)
                for j, gradient in enumerate(gradients):
                    first[j] = 0.9 * first[j] + 0.1 * gradient
                    second[j] = 0.999 * second[j] + 0.001 * gradient**2
                    self.parameters[j] -= (
                        0.008
                        * (first[j] / (1 - 0.9**step))
                        / (np.sqrt(second[j] / (1 - 0.999**step)) + 1e-8)
                    )
            self.losses.append(float(np.mean(losses)))
        wu, bu, wi, bi = self.parameters
        self.user_embeddings = np.maximum(self.user_features @ wu + bu, 0)
        self.item_embeddings = np.maximum(self.item_features @ wi + bi, 0)
        return self

    def scores(self, user):
        return (
            self.item_embeddings
            @ self.user_embeddings[user]
            / np.sqrt(self.item_embeddings.shape[1])
        )


class MatrixFactorisation:
    def __init__(self, n_users, n_items, seed=17, dimensions=24):
        rng = np.random.default_rng(seed)
        self.users = rng.normal(0, 0.05, (n_users, dimensions))
        self.items = rng.normal(0, 0.05, (n_items, dimensions))
        self.bias = np.zeros(n_items)
        self.losses = []

    def fit(self, positives, seen, catalog, epochs=12, seed=18):
        rng = np.random.default_rng(seed)
        u, p, n = triples(positives, seen, catalog, rng)
        for _ in range(epochs):
            losses = []
            for batch in np.array_split(rng.permutation(len(u)), max(1, len(u) // 256)):
                ub, pb, nb = u[batch], p[batch], n[batch]
                a = self.users[ub].copy()
                b = self.items[pb].copy()
                c = self.items[nb].copy()
                diff = np.sum(a * (b - c), axis=1) + self.bias[pb] - self.bias[nb]
                g = 1 / (1 + np.exp(np.clip(diff, -40, 40)))
                np.add.at(self.users, ub, 0.035 * (g[:, None] * (b - c) - 0.005 * a))
                np.add.at(self.items, pb, 0.035 * (g[:, None] * a - 0.005 * b))
                np.add.at(self.items, nb, 0.035 * (-g[:, None] * a - 0.005 * c))
                np.add.at(self.bias, pb, 0.035 * g)
                np.add.at(self.bias, nb, -0.035 * g)
                losses.append(np.logaddexp(0, -diff).mean())
            self.losses.append(float(np.mean(losses)))
        self.warm_items = np.bincount(positives[:, 1], minlength=len(self.items)) > 0
        return self

    def scores(self, user):
        scores = self.items @ self.users[user] + self.bias
        scores[~self.warm_items] = 0
        return scores
