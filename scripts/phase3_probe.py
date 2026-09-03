"""Phase 3 -- train the truth probe, and run the confound control (design doc s3).

No GPU: everything reads cached activations written by `phase1_harness.py`.

    python scripts/phase3_probe.py --acts data/phase2/acts_v2

**3.1 Split.** Doc's rule: train on debates where steering failed or was not
attempted (the judge got it right, so truth and verdict agree); test on the
successfully-steered ones, where they diverge. The adversarial distribution stays
fully held out so the probe can never learn "this transcript looks manipulated".

Layered on top of that, and decided before any of these numbers existed: **no article
may appear in both halves.** With two questions per article, the doc's outcome-based
split would otherwise put a story's sibling debate in training whenever one of its
debates landed in test, and the probe could win by recognising the story. Test-side
articles are therefore excluded from training wholesale.

**3.2 Truth probe.** Mass-mean over contrast pairs, labelled by the *gold* answer.
Layer chosen on a validation slice of the training set -- never on test, and never by
argmax over the test set, which is the mistake Phase 0 flagged when Gate C's best layer
turned out to be an isolated spike among 64.

**3.3 Verdict probe -- the load-bearing control.** A second probe labelled by the
judge's *actual verdict* rather than by gold. If the two directions are near-identical,
the "truth probe" is only recovering the verdict and Q1 collapses; that is a real
negative and gets reported. If they are separable, there is a representation distinct
from the output and Q1 is live. Reported as cosine plus both cross-tests.

Activations are the mean over the two presentation orders, which is what makes each
debate one order-invariant example (Phase 0's counterbalancing decision).

**Two test sets, both fixed in advance** (`--criterion`). `verdict` is primary: a
debate is steered when both presentation orders agree on the distractor. `forced` is
the pre-registered secondary: the sign of the order-averaged P(gold), which is
defined for every debate, so it neither conditions on order-consistency -- a
post-treatment variable the Q0 entry flags as affected by the treatment -- nor
discards the debates presentation order decided. They yield 23 and 80 test cases
respectively, and the forced criterion also gives the 3.3 control far more
gold-vs-verdict disagreements to sit on. Both are reported; neither was chosen after
seeing which flatters the probe.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from judgeprobe import config  # noqa: E402
from judgeprobe.analysis import (  # noqa: E402
    article_val_split, criterion_maps, split_by_outcome, wilson,
)
from judgeprobe.probes import (  # noqa: E402
    accuracy, best_band, best_layer, cosine, fit_logistic, fit_mass_mean,
    label_layers, loo_accuracy, sweep_layers,
)
from judgeprobe.store import ActivationStore  # noqa: E402
from judgeprobe.transcripts import read_jsonl  # noqa: E402


def order_mean(store: ActivationStore, tid: str, cls: str) -> np.ndarray:
    """Mean activation across the two presentation orders, for one contrast class."""
    fwd = store.load(tid, f"full__forward__{cls}")
    rev = store.load(tid, f"full__reverse__{cls}")
    return (fwd + rev) / 2.0


def build(store, ids, condition_of, article_of, verdict_of):
    """Return (gold_acts, distractor_acts, meta) aligned by debate."""
    pos, neg, meta = [], [], []
    for tid in ids:
        pos.append(order_mean(store, tid, "gold"))
        neg.append(order_mean(store, tid, "distractor"))
        meta.append({"id": tid, "article": article_of[tid],
                     "condition": condition_of[tid], "verdict": verdict_of[tid]})
    return np.stack(pos), np.stack(neg), meta


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--acts", type=Path, default=config.PHASE2_DIR / "acts_v2")
    ap.add_argument("--transcripts", type=Path,
                    default=config.PHASE2_DIR / "transcripts.jsonl")
    ap.add_argument("--val-frac", type=float, default=0.25,
                    help="fraction of training ARTICLES held out for layer selection")
    ap.add_argument("--criterion", choices=("verdict", "forced"), default="verdict",
                    help="what counts as a steered debate: both orders agree on the "
                         "distractor (primary), or order-averaged P(gold) < 0.5 "
                         "(pre-registered secondary)")
    ap.add_argument("--seed", type=int, default=config.SEED)
    ap.add_argument("--out", type=Path, default=None,
                    help="report path; defaults to <acts>/phase3_report[_criterion].json")
    args = ap.parse_args()

    transcripts = read_jsonl(args.transcripts)
    condition_of = {t.transcript_id: t.condition for t in transcripts}
    article_of = {t.transcript_id: t.meta["article_id"] for t in transcripts}

    store = ActivationStore(args.acts)
    rows = {r["transcript_id"]: r for r in
            (i["verdict"] for i in store.info_table() if "verdict" in i)}
    verdict_of = {k: v for k, v in rows.items()}
    ids = [k for k in store.keys if k in condition_of and k in rows]
    print(f"{len(ids)} debates with activations and verdicts\n")

    # --- 3.1 split ----------------------------------------------------------
    # Which debates count as decided, and what the judge chose on the decided ones.
    # Under `verdict` a debate is decided only if the two presentation orders agree;
    # under `forced` every debate is decided, by the sign of the order-averaged
    # P(gold). Everything downstream -- test set, training set, control set -- reads
    # these two maps, so the criterion is applied in exactly one place.
    decided, chose_gold = criterion_maps(ids, rows, args.criterion)
    steered, train, excluded, test_articles = split_by_outcome(
        ids, condition_of=condition_of, article_of=article_of,
        decided=decided, chose_gold=chose_gold)

    print(f"3.1 split   (criterion: {args.criterion})")
    print(f"  test  (successfully steered)      : {len(steered)}")
    print(f"  train (judge chose gold)          : {len(train)}  "
          f"[{sum(1 for i in train if condition_of[i] == 'honest')} honest, "
          f"{sum(1 for i in train if condition_of[i] == 'collusion')} failed-steering]")
    print(f"  excluded to keep articles disjoint: {len(excluded)}")
    print(f"  train articles {len({article_of[i] for i in train})}, "
          f"test articles {len(test_articles)}, "
          f"overlap {len({article_of[i] for i in train} & test_articles)}")
    if len(steered) < 5 or len(train) < 20:
        print("\nnot enough data to fit; stopping")
        return 1

    # Validation split by ARTICLE, for layer selection only.
    fit_ids, val_ids, val_articles = article_val_split(
        train, article_of, val_frac=args.val_frac, seed=args.seed)
    print(f"  layer-selection split: fit {len(fit_ids)}, val {len(val_ids)} "
          f"({len(val_articles)} articles held out)\n")

    P_fit, N_fit, _ = build(store, fit_ids, condition_of, article_of, verdict_of)
    P_val, N_val, _ = build(store, val_ids, condition_of, article_of, verdict_of)
    P_test, N_test, meta_test = build(store, steered, condition_of, article_of, verdict_of)

    # --- 3.2 truth probe ----------------------------------------------------
    print("3.2 truth probe (labels = gold answer)")
    sweep = []
    for layer in range(P_fit.shape[1]):
        pr = fit_mass_mean(P_fit[:, layer, :], N_fit[:, layer, :], layer=layer)
        sweep.append({"layer": layer,
                      "val_acc": accuracy(pr, P_val[:, layer, :], N_val[:, layer, :]),
                      "train_acc": accuracy(pr, P_fit[:, layer, :], N_fit[:, layer, :])})
    sweep = label_layers(sweep, store.manifest.get("layers"))
    best = max(sweep, key=lambda r: r["val_acc"])
    band = best_band(sweep, key="val_acc")
    chosen = band["layer"] if band else best["layer"]
    chosen_row = next(r for r in sweep if r["layer"] == chosen)
    print(f"  best single layer : {best.get('model_layer', best['layer'])} "
          f"val acc {best['val_acc']:.3f}")
    print(f"  band centre       : {chosen_row.get('model_layer', chosen)} "
          f"val acc {chosen_row['val_acc']:.3f}   <- used")
    above = sum(1 for r in sweep if r["val_acc"] >= 0.75)
    print(f"  layers >= 0.75 val: {above}/{len(sweep)}")

    truth = fit_mass_mean(P_fit[:, chosen, :], N_fit[:, chosen, :], layer=chosen)
    train_acc = accuracy(truth, P_fit[:, chosen, :], N_fit[:, chosen, :])
    val_acc = accuracy(truth, P_val[:, chosen, :], N_val[:, chosen, :])

    # --- 3.3 verdict probe (the control) -----------------------------------
    # The control is only meaningful on debates where gold and verdict DISAGREE.
    # Fitting it on the 3.1 training split alone is vacuous: that split is "judge
    # got it right", so the two labellings are identical there and cosine is 1.000
    # by construction rather than by finding. So the control set is every
    # order-consistent debate outside the test articles -- judge-correct and
    # judge-wrong alike -- which is the only place the two directions can diverge.
    print("\n3.3 verdict probe (labels = judge's verdict) -- the control")
    ctrl_ids = [i for i in ids
                if i not in steered
                and article_of[i] not in test_articles
                and decided[i]]
    n_disagree = sum(1 for i in ctrl_ids if not chose_gold[i])
    print(f"  control set: {len(ctrl_ids)} decided debates, "
          f"{n_disagree} of them with gold != verdict")

    gp, gn, vp, vn = [], [], [], []
    for tid in ctrl_ids:
        g = order_mean(store, tid, "gold")
        d = order_mean(store, tid, "distractor")
        gp.append(g); gn.append(d)                      # labelled by gold
        vp.append(g if chose_gold[tid] else d)          # labelled by verdict
        vn.append(d if chose_gold[tid] else g)
    GP, GN = np.stack(gp), np.stack(gn)
    VP, VN = np.stack(vp), np.stack(vn)

    truth_c = fit_mass_mean(GP[:, chosen, :], GN[:, chosen, :], layer=chosen)
    verdict = fit_mass_mean(VP[:, chosen, :], VN[:, chosen, :], layer=chosen)

    cos = cosine(truth_c, verdict)
    truth_on_verdict = accuracy(truth_c, VP[:, chosen, :], VN[:, chosen, :])
    verdict_on_truth = accuracy(verdict, GP[:, chosen, :], GN[:, chosen, :])
    print(f"  cosine(truth, verdict)      : {cos:.3f}")
    print(f"  truth probe on verdict task : {truth_on_verdict:.3f}")
    print(f"  verdict probe on truth task : {verdict_on_truth:.3f}")
    if n_disagree < 10:
        print(f"  -> WEAK: only {n_disagree} disagreement cases, so the two "
              f"labellings barely differ. Treat the cosine as uninformative.")
    elif cos > 0.9:
        print("  -> directions near-identical: the truth probe may only be "
              "recovering the verdict. Q1 collapses (pre-registered negative #1).")
    else:
        print("  -> directions separable: a representation distinct from the "
              "output. Q1 is live.")

    # --- 4. Q1 on the steered set ------------------------------------------
    print("\nQ1 -- probe vs judge on the held-out steered set")
    scores_gold = P_test[:, chosen, :] @ truth.direction
    scores_dist = N_test[:, chosen, :] @ truth.direction
    correct = int(np.sum(scores_gold > scores_dist))
    n = len(steered)
    lo, hi = wilson(correct, n)
    print(f"  judge accuracy : 0.000  ({n} debates, wrong by construction)")
    print(f"  probe accuracy : {correct/n:.3f}  [{correct}/{n}]  95% CI [{lo:.3f}, {hi:.3f}]")
    print(f"  train acc {train_acc:.3f} | val acc {val_acc:.3f}")

    from math import comb
    p_above = sum(comb(n, i) for i in range(correct, n + 1)) / 2 ** n
    p_below = sum(comb(n, i) for i in range(0, correct + 1)) / 2 ** n
    print(f"  binomial vs chance: p(above) = {p_above:.4f}   "
          f"p(below) = {p_below:.6f}")
    if p_below < 0.05:
        print("  -> the probe is significantly ANTI-correlated with truth here: on "
              "steered debates it points at the answer the judge chose, not the gold "
              "one. That is the verdict direction, not a truth representation.")

    # Does the probe simply reproduce the judge? On this set the judge always chose
    # the distractor, so "probe agrees with judge" is 1 - probe accuracy.
    print(f"  probe agrees with the judge's (wrong) verdict on "
          f"{n - correct}/{n} = {(n - correct)/n:.3f}")

    # --- diagnostics, and they are labelled as diagnostics ------------------
    # Test-set accuracy at every layer. This is selection on the test set, so its
    # maximum is NOT a result and never picks the layer -- `chosen` is already fixed
    # by the validation articles above. It answers the one question a reader will
    # ask of a negative: is the truth kept at some other depth, or is the verdict
    # direction the whole story everywhere?
    test_curve = []
    for layer in range(P_fit.shape[1]):
        pr = fit_mass_mean(P_fit[:, layer, :], N_fit[:, layer, :], layer=layer)
        sg = P_test[:, layer, :] @ pr.direction
        sd = N_test[:, layer, :] @ pr.direction
        test_curve.append({"layer": layer, "test_acc": float(np.mean(sg > sd))})
    test_curve = label_layers(test_curve, store.manifest.get("layers"))
    best_test = max(test_curve, key=lambda r: r["test_acc"])
    n_above_chance = sum(1 for r in test_curve if r["test_acc"] > 0.5)
    print(f"\ndiagnostic (selection on test -- not a result)")
    print(f"  best layer on test: {best_test.get('model_layer', best_test['layer'])} "
          f"at {best_test['test_acc']:.3f}")
    print(f"  layers above chance on test: {n_above_chance}/{len(test_curve)}")

    # Logistic as the doc's secondary comparison (s0.5), at the chosen layer only:
    # does the negative depend on mass-mean, or on the activations?
    log_probe = fit_logistic(P_fit[:, chosen, :], N_fit[:, chosen, :], layer=chosen)
    lg = P_test[:, chosen, :] @ log_probe.direction
    ld = N_test[:, chosen, :] @ log_probe.direction
    logistic_test_acc = float(np.mean(lg > ld))
    logistic_val_acc = accuracy(log_probe, P_val[:, chosen, :], N_val[:, chosen, :])
    cos_mass_logistic = cosine(truth, log_probe)
    print(f"  logistic (secondary): val {logistic_val_acc:.3f}, "
          f"test {logistic_test_acc:.3f}, cosine with mass-mean {cos_mass_logistic:.3f}")

    report = {
        "criterion": args.criterion,
        "train_honest": sum(1 for i in train if condition_of[i] == "honest"),
        "train_failed_steering": sum(1 for i in train
                                     if condition_of[i] == "collusion"),
        "test_curve_diagnostic": test_curve,
        "best_test_layer": best_test.get("model_layer", best_test["layer"]),
        "best_test_acc": best_test["test_acc"],
        "test_layers_above_chance": n_above_chance,
        "logistic_val_acc": logistic_val_acc,
        "logistic_test_acc": logistic_test_acc,
        "cosine_mass_logistic": cos_mass_logistic,
        "n_test": n, "n_train": len(train), "n_excluded": len(excluded),
        "layer": int(chosen),
        "model_layer": chosen_row.get("model_layer", int(chosen)),
        "train_acc": train_acc, "val_acc": val_acc,
        "probe_correct": correct, "probe_acc": correct / n,
        "probe_ci95": [lo, hi],
        "binomial_p_above_chance": p_above,
        "binomial_p_below_chance": p_below,
        "probe_agrees_with_judge": (n - correct) / n,
        "control_n": len(ctrl_ids), "control_n_disagree": n_disagree,
        "judge_acc": 0.0,
        "cosine_truth_verdict": cos,
        "truth_on_verdict": truth_on_verdict,
        "verdict_on_truth": verdict_on_truth,
        "layers_above_0.75_val": above,
        "sweep": sweep,
        "test_ids": [m["id"] for m in meta_test],
    }
    suffix = "" if args.criterion == "verdict" else f"_{args.criterion}"
    out = args.out or (args.acts / f"phase3_report{suffix}.json")
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nreport -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
