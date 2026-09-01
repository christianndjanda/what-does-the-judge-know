"""QuALITY loading, restricted to the two-option setting: gold vs best distractor.

QuALITY stores one record per article; each record holds several questions.
`gold_label` and each annotator's `untimed_best_distractor` are 1-indexed into
`options`.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterator

from . import config

SPLITS = ("train", "dev", "test")


@dataclass(frozen=True)
class QualityQuestion:
    """One QuALITY question reduced to the two options this project uses."""

    question_id: str
    article_id: str
    title: str
    article: str
    question: str
    gold: str
    best_distractor: str
    gold_index: int  # 1-indexed into the original 4 options
    best_distractor_index: int
    difficult: bool
    n_annotators: int
    distractor_votes: int  # how many annotators picked that distractor

    def to_dict(self, include_article: bool = False) -> dict:
        d = asdict(self)
        if not include_article:
            d.pop("article")
        return d


def _split_path(split: str, html_stripped: bool = True) -> Path:
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}, got {split!r}")
    stem = "QuALITY.v1.0.1.htmlstripped" if html_stripped else "QuALITY.v1.0.1"
    return config.quality_dir() / f"{stem}.{split}"


def best_distractor_for_question(question: dict) -> tuple[int, int]:
    """Return (1-indexed option, n_votes) for the annotators' best distractor.

    Annotators each name the option they consider the most tempting wrong answer.
    We take the plurality vote, excluding any vote that lands on the gold answer
    (annotators occasionally mark the gold option, which is not a distractor).
    Ties break toward the lowest option index for determinism.
    """
    gold = question["gold_label"]
    votes = Counter(
        v["untimed_best_distractor"]
        for v in question.get("validation", [])
        if v.get("untimed_best_distractor") not in (None, gold)
    )
    if not votes:
        raise ValueError(f"no usable best-distractor votes for {question['question_unique_id']}")
    top = max(votes.values())
    winner = min(opt for opt, n in votes.items() if n == top)
    return winner, votes[winner]


def iter_questions(
    split: str = "dev",
    *,
    require_unanimous_gold: bool = True,
    min_distractor_votes: int = 2,
) -> Iterator[QualityQuestion]:
    """Yield two-option questions from a split.

    `require_unanimous_gold` keeps only questions where every untimed annotator
    agreed with the gold label. The judge's errors need to come from the debate,
    not from genuine question ambiguity.
    """
    path = _split_path(split)
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            article = json.loads(line)
            for q in article["questions"]:
                validation = q.get("validation", [])
                if not validation:
                    continue
                gold = q.get("gold_label")
                if gold is None:
                    continue
                if require_unanimous_gold and any(
                    v.get("untimed_answer") != gold for v in validation
                ):
                    continue
                try:
                    distractor, votes = best_distractor_for_question(q)
                except ValueError:
                    continue
                if votes < min_distractor_votes:
                    continue
                yield QualityQuestion(
                    question_id=q["question_unique_id"],
                    article_id=str(article["article_id"]),
                    title=article["title"],
                    article=article["article"],
                    question=q["question"].strip(),
                    gold=q["options"][gold - 1].strip(),
                    best_distractor=q["options"][distractor - 1].strip(),
                    gold_index=gold,
                    best_distractor_index=distractor,
                    difficult=bool(q.get("difficult", 0)),
                    n_annotators=len(validation),
                    distractor_votes=votes,
                )


def sample_questions(n: int, split: str = "dev", seed: int = config.SEED, **kw) -> list[QualityQuestion]:
    """Deterministically sample `n` questions, at most one per article.

    Superseded by `sample_by_article` for Phase 2, and kept because the first
    40-debate corpus was drawn with it.
    """
    import random

    by_article: dict[str, list[QualityQuestion]] = {}
    for q in iter_questions(split, **kw):
        by_article.setdefault(q.article_id, []).append(q)
    rng = random.Random(seed)
    article_ids = sorted(by_article)
    rng.shuffle(article_ids)
    return [rng.choice(by_article[a]) for a in article_ids[:n]]


def articles_available(splits=("dev", "train"), **kw) -> int:
    """How many distinct articles pass the filter -- the ceiling on `n_articles`."""
    return len({q.article_id for s in splits for q in iter_questions(s, **kw)})


def sample_by_article(
    n_articles: int,
    *,
    splits=("dev", "train"),
    per_article: int = 1,
    seed: int = config.SEED,
    **kw,
) -> list[QualityQuestion]:
    """Sample `n_articles` articles and `per_article` questions from each.

    `sample_questions` takes one question per article, which caps a corpus at the
    number of articles: 115 on dev, 149 on train. Reaching a useful number of
    *steered* debates needs more than that (see logbook), and QuALITY has ~12
    eligible questions per article going unused.

    Two properties this guarantees, both load-bearing:

    * **Deterministic and extensible in both directions.** Articles are drawn from a
      seeded shuffle and questions within an article are taken in `question_id` order,
      so raising `per_article` or `n_articles` *appends* to the previous selection
      instead of reshuffling it. A corpus built at k=2 extends to k=4 by generating
      the two new questions per article; the existing debates keep their ids and are
      skipped by resume.
    * **Article grouping survives.** Callers get a flat list, but every question
      carries `article_id`, which is what lets `plan_corpus` keep whole articles on
      one side of the honest/collusion split. With one question per article that was
      automatic; with several it has to be enforced, or the same story lands in both
      the probe's training and test sets.

    `test` is excluded from the default splits deliberately: public QuALITY test ships
    without gold labels.
    """
    import random

    by_article: dict[str, list[QualityQuestion]] = {}
    for split in splits:
        for q in iter_questions(split, **kw):
            by_article.setdefault(q.article_id, []).append(q)

    article_ids = sorted(by_article)
    random.Random(seed).shuffle(article_ids)

    out: list[QualityQuestion] = []
    for aid in article_ids[:n_articles]:
        questions = sorted(by_article[aid], key=lambda q: q.question_id)
        out.extend(questions[:per_article])
    return out
