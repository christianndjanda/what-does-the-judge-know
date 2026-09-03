"""Does the judge know the answer on the debates presentation order decided?

**Exploratory, and post-hoc.** This test set was not pre-registered. It exists because
the two pre-registered ones disagreed in a way that pointed at it: on the primary
(verdict) criterion the probe followed the judge's wrong answer 22/23 times, while on
the secondary (forced) criterion it read *gold* on 32 of the 43 test debates where the
judge had barely committed. Reported as a hypothesis this data suggests, not as a
result it establishes -- the honest version of that distinction is that someone should
run it again on a fresh corpus.

Two flaws in reading that 32/43 straight off the forced run, both fixed here:

1. **Two different probes.** The forced run selects its own layer on its own training
   split (47, val 0.810); the primary run selects 44 at val 0.912. Comparing bins
   across the two compares probes. Here there is one probe: the primary one.
2. **Selection on the outcome.** The forced criterion's test set is the debates whose
   order-averaged P(gold) fell *below* 0.5. Among near-ties that is a coin flip, so
   conditioning on it selects half of a cluster on noise. Here the test set is **every
   order-split debate**, whichever side of 0.5 it landed -- no outcome selection at
   all.

An order-split debate is one the two presentation orders answered differently: the
judge picked the same *slot* both times, so its output was decided by layout rather
than by the debate. 342 of the 525 debates are like this, and Q0 showed collusion
makes them commoner (0.72 against 0.58). The judge's own accuracy on them is a coin
flip by construction. The question is whether the residual stream is.

Article-disjointness is enforced against the probe's training articles, so a debate
here never shares a story with one the probe was fitted on.

    python scripts/phase3_crosstest.py --acts data/phase2/acts_rounds
"""

from __future__ import annotations

import argparse
import json
import sys
from math import comb
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from judgeprobe import config  # noqa: E402
from judgeprobe.analysis import (  # noqa: E402
    article_val_split, criterion_maps, split_by_outcome, wilson,
)
from judgeprobe.probes import accuracy, best_band, fit_mass_mean, label_layers  # noqa: E402
from judgeprobe.store import ActivationStore  # noqa: E402
from judgeprobe.transcripts import read_jsonl  # noqa: E402


def order_mean(store, tid, cls):
    return (store.load(tid, f"full__forward__{cls}")
            + store.load(tid, f"full__reverse__{cls}")) / 2.0


def p_above(k: int, n: int) -> float:
    return sum(comb(n, i) for i in range(k, n + 1)) / 2 ** n


def score(store, ids, layer, direction):
    """`(k_gold, n)` -- how often the probe prefers the gold answer."""
    if not ids:
        return 0, 0
    g = np.stack([order_mean(store, i, "gold")[layer] for i in ids]) @ direction
    d = np.stack([order_mean(store, i, "distractor")[layer] for i in ids]) @ direction
    return int(np.sum(g > d)), len(ids)


def report_line(name, k, n, judge_k=None):
    if n == 0:
        print(f"  {name:34s}      -- no debates")
        return None
    lo, hi = wilson(k, n)
    p = p_above(k, n)
    row = {"set": name, "n": n, "probe_correct": k, "probe_acc": k / n,
           "ci95": [lo, hi], "binomial_p_above_chance": p}
    extra = ""
    if judge_k is not None:
        row["judge_acc"] = judge_k / n
        extra = f"   judge {judge_k / n:.3f}   gap {(k - judge_k) / n:+.3f}"
    print(f"  {name:34s} n={n:4d}  probe {k/n:.3f} "
          f"[{lo:.3f}, {hi:.3f}]  p={p:.4f}{extra}")
    return row


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--acts", type=Path, default=config.PHASE2_DIR / "acts_rounds")
    ap.add_argument("--transcripts", type=Path,
                    default=config.PHASE2_DIR / "transcripts.jsonl")
    ap.add_argument("--val-frac", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=config.SEED)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    transcripts = read_jsonl(args.transcripts)
    condition_of = {t.transcript_id: t.condition for t in transcripts}
    article_of = {t.transcript_id: t.meta["article_id"] for t in transcripts}

    store = ActivationStore(args.acts)
    rows = {r["transcript_id"]: r for r in
            (i["verdict"] for i in store.info_table() if "verdict" in i)}
    ids = [k for k in store.keys if k in condition_of and k in rows]

    # The primary probe, exactly as Phase 3 builds it.
    decided, chose_gold = criterion_maps(ids, rows, "verdict")
    steered, train, _, test_articles = split_by_outcome(
        ids, condition_of=condition_of, article_of=article_of,
        decided=decided, chose_gold=chose_gold)
    fit_ids, val_ids, _ = article_val_split(train, article_of,
                                            val_frac=args.val_frac, seed=args.seed)
    P_fit = np.stack([order_mean(store, i, "gold") for i in fit_ids])
    N_fit = np.stack([order_mean(store, i, "distractor") for i in fit_ids])
    P_val = np.stack([order_mean(store, i, "gold") for i in val_ids])
    N_val = np.stack([order_mean(store, i, "distractor") for i in val_ids])

    sweep = []
    for layer in range(P_fit.shape[1]):
        pr = fit_mass_mean(P_fit[:, layer, :], N_fit[:, layer, :], layer=layer)
        sweep.append({"layer": layer,
                      "val_acc": accuracy(pr, P_val[:, layer, :], N_val[:, layer, :]),
                      "train_acc": accuracy(pr, P_fit[:, layer, :], N_fit[:, layer, :])})
    sweep = label_layers(sweep, store.manifest.get("layers"))
    band = best_band(sweep, key="val_acc")
    chosen = band["layer"] if band else max(sweep, key=lambda r: r["val_acc"])["layer"]
    chosen_row = next(r for r in sweep if r["layer"] == chosen)
    model_layer = chosen_row.get("model_layer", chosen)
    probe = fit_mass_mean(P_fit[:, chosen, :], N_fit[:, chosen, :], layer=chosen)
    print(f"probe: model layer {model_layer}, fitted on {len(fit_ids)} debates, "
          f"val acc {chosen_row['val_acc']:.3f}")

    # The test set: order-split debates on articles the probe never saw.
    fit_articles = {article_of[i] for i in fit_ids} | {article_of[i] for i in val_ids}
    split_ids = [i for i in ids
                 if not decided[i]
                 and article_of[i] not in fit_articles
                 and article_of[i] not in test_articles]
    print(f"order-split debates: {sum(1 for i in ids if not decided[i])} total, "
          f"{len(split_ids)} on articles disjoint from the probe's training\n")

    results = []
    print("probe vs judge where presentation order decided the output")
    judge_k = sum(1 for i in split_ids if rows[i]["forced_choice"] == "gold")
    results.append(report_line("all order-split", *score(store, split_ids, chosen,
                                                         probe.direction), judge_k))
    for cond in ("collusion", "honest"):
        sub = [i for i in split_ids if condition_of[i] == cond]
        jk = sum(1 for i in sub if rows[i]["forced_choice"] == "gold")
        results.append(report_line(f"  {cond}", *score(store, sub, chosen,
                                                       probe.direction), jk))
    near = [i for i in split_ids if abs(rows[i]["p_gold"] - 0.5) < 0.05]
    results.append(report_line("  of those, near-ties (<0.05)",
                               *score(store, near, chosen, probe.direction),
                               sum(1 for i in near
                                   if rows[i]["forced_choice"] == "gold")))

    print("\nanchor: the pre-registered primary test set, same probe")
    results.append(report_line("confident, rhetorically steered",
                               *score(store, steered, chosen, probe.direction), 0))

    print("\n  'judge' is the judge's forced-choice accuracy -- the soft readout, "
          "which keeps signal on these debates even though the hard verdict does not. "
          "The verdict itself is decided by the slot here: with option letters "
          "balanced, it is right about half the time in expectation, whichever way "
          "the debate went. So the gap column is the interesting one -- probe against "
          "the strongest thing the judge's own output can offer, not against the "
          "answer it actually gave. Above chance beats the verdict; above the judge "
          "column means the residual stream carries more than the logits do.")

    report = {"model_layer": int(model_layer), "val_acc": chosen_row["val_acc"],
              "n_fit": len(fit_ids), "n_order_split_total": sum(1 for i in ids
                                                                if not decided[i]),
              "n_order_split_disjoint": len(split_ids),
              "results": [r for r in results if r]}
    out = args.out or (args.acts / "phase3_crosstest.json")
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nreport -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
