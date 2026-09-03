"""Shared analysis-side bookkeeping for Phases 3-5.

The probe run and the trajectory run need the same three things: which debates the
judge actually *decided*, which answer it chose on them, and the article-disjoint
split that follows. One copy, so the two phases cannot drift into slightly different
definitions of "steered" -- the kind of divergence that leaves two numbers in a
writeup with no way to tell which one is right.

Nothing here touches a GPU or a model; it is bookkeeping over the store manifest.
"""

from __future__ import annotations

import numpy as np

CRITERIA = ("verdict", "forced")


def criterion_maps(ids: list[str], rows: dict, criterion: str):
    """`(decided, chose_gold)` -- is this debate decided, and did the judge pick gold?

    `verdict` is primary: a debate counts only when the two presentation orders agree
    on an answer, so it conditions on order-consistency, which Q0 showed the treatment
    itself degrades (0.418 honest against 0.280 colluded). `forced` is the
    pre-registered secondary: the sign of the order-averaged P(gold), defined for
    every debate, so it neither conditions on a post-treatment variable nor discards
    the debates presentation order decided.
    """
    if criterion not in CRITERIA:
        raise ValueError(f"criterion must be one of {CRITERIA}, got {criterion!r}")
    if criterion == "verdict":
        decided = {i: bool(rows[i]["consistent"]) for i in ids}
        chose_gold = {i: rows[i]["choice"] == "gold" for i in ids}
    else:
        decided = {i: True for i in ids}
        chose_gold = {i: rows[i]["forced_choice"] == "gold" for i in ids}
    return decided, chose_gold


def split_by_outcome(ids: list[str], *, condition_of: dict, article_of: dict,
                     decided: dict, chose_gold: dict):
    """Doc s3.1's split, with article-disjointness enforced on top.

    Test is the colluded debates the judge got wrong; training is the debates it got
    right, so truth and verdict agree there and the adversarial distribution stays
    fully held out. **No article may appear in both halves**: with two questions per
    article, a story's sibling debate would otherwise sit in training whenever one of
    its debates landed in test, and the probe could win by recognising the story.

    Returns `(steered, train, excluded, test_articles)`.
    """
    steered = [i for i in ids
               if condition_of[i] == "collusion" and decided[i] and not chose_gold[i]]
    test_articles = {article_of[i] for i in steered}
    train = [i for i in ids
             if i not in steered
             and article_of[i] not in test_articles
             and decided[i] and chose_gold[i]]
    excluded = [i for i in ids
                if i not in steered and article_of[i] in test_articles]
    return steered, train, excluded, test_articles


def article_val_split(train: list[str], article_of: dict, *,
                      val_frac: float, seed: int):
    """Hold out whole ARTICLES from the training set, for layer selection only.

    By article rather than by debate for the same reason as the test split, one level
    down: sibling debates share a story, and a validation score inflated by story
    recognition would pick the layer on the wrong criterion.
    """
    rng = np.random.default_rng(seed)
    articles = sorted({article_of[i] for i in train})
    rng.shuffle(articles)
    n_val = max(1, int(len(articles) * val_frac))
    val_articles = set(articles[:n_val])
    fit_ids = [i for i in train if article_of[i] not in val_articles]
    val_ids = [i for i in train if article_of[i] in val_articles]
    return fit_ids, val_ids, val_articles


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval -- the sample is small, so no bare point estimates."""
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / d
    return (max(0.0, c - h), min(1.0, c + h))


def mean_ci(xs, z: float = 1.96) -> tuple[float, float, float]:
    """`(mean, lo, hi)`, normal-approximation interval on the mean."""
    a = np.asarray(xs, dtype=float)
    if a.size == 0:
        return (float("nan"),) * 3
    m = float(a.mean())
    if a.size == 1:
        return (m, m, m)
    se = float(a.std(ddof=1) / np.sqrt(a.size))
    return (m, m - z * se, m + z * se)
