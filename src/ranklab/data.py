from datetime import datetime, timezone
import hashlib, json, zipfile
from pathlib import Path
import numpy as np


def load(path, expected_sha256=None):
    path = Path(path)
    if (
        expected_sha256
        and hashlib.sha256(path.read_bytes()).hexdigest() != expected_sha256
    ):
        raise ValueError("MovieLens fingerprint mismatch")
    with zipfile.ZipFile(path) as archive:
        ratings = np.loadtxt(archive.open("ml-100k/u.data"), dtype=np.int64)
        genres = []
        released = []
        for line in archive.read("ml-100k/u.item").decode("latin-1").splitlines():
            fields = line.split("|")
            genres.append([int(v) for v in fields[5:]])
            try:
                date = (
                    datetime.strptime(fields[2], "%d-%b-%Y")
                    .replace(tzinfo=timezone.utc)
                    .timestamp()
                )
            except ValueError:
                date = float("inf")
            released.append(date)
    ratings[:, :2] -= 1
    return ratings, np.array(genres, float), np.array(released)


def split(ratings):
    if ratings.ndim != 2 or ratings.shape[1] != 4:
        raise ValueError("Expected user,item,rating,timestamp rows")
    times = ratings[:, 3]
    a, b = np.quantile(times, [0.70, 0.85]).astype(np.int64)
    return (
        ratings[times < a],
        ratings[(times >= a) & (times < b)],
        ratings[times >= b],
        int(a),
        int(b),
    )


def training_features(train, genres, n_users):
    positives = train[train[:, 2] >= 4]
    user = np.zeros((n_users, genres.shape[1] + 1))
    counts = np.zeros(n_users)
    np.add.at(user[:, :-1], positives[:, 0], genres[positives[:, 1]])
    np.add.at(counts, positives[:, 0], 1)
    user[:, :-1] /= np.maximum(counts[:, None], 1)
    user[:, -1] = np.log1p(counts) / 5
    popularity = np.bincount(positives[:, 1], minlength=len(genres)).astype(float)
    item = np.c_[genres, np.log1p(popularity) / 6]
    seen = [set() for _ in range(n_users)]
    for u, i, _, _ in train:
        seen[u].add(int(i))
    return positives, user, item, counts, popularity, seen


def triples(positives, seen, catalog, rng):
    negative = np.empty(len(positives), int)
    for index, (user, _, _, _) in enumerate(positives):
        # A dense seen history must not create an infinite rejection loop.
        allowed = np.array([item for item in catalog if item not in seen[user]])
        if not len(allowed):
            raise ValueError("User has no unseen catalogue items")
        negative[index] = rng.choice(allowed)
    return positives[:, 0], positives[:, 1], negative
