"""Phase 5 -- Q2, the probe's trajectory across debate rounds (design doc s5).

No GPU: reads the round-truncated activation store written by

    python scripts/phase1_harness.py --model <id> --rounds 1,2,full \
        --transcripts data/phase2/transcripts.jsonl --out data/phase2/acts_rounds

then

    python scripts/phase5_trajectory.py --acts data/phase2/acts_rounds

**The probe is trained once and applied at every round**, per the doc: fit on the
*full* transcripts of the training split, then scored against each truncation. A
probe refitted per round would answer a different question -- "is the answer
recoverable at round k" rather than "what happens to the direction we found".

**What Q2 now measures.** As posed, Q2 asks whether the gold-answer signal stays flat
(steering overrides the output) or decays (steering corrupts the representation).
Phase 3 found the direction is the *verdict* direction, not a truth representation,
so the honest reading of these curves is **when the judge commits** -- how early in a
debate the residual stream settles on the answer it will give. Reported under that
caption, not the doc's original one.

**Length confound control (pre-registered, required).** Longer context may move probe
scores independent of content, so the identical analysis runs on honest debates the
judge got right, and both curves are reported on the same axes. The control debates
come from the held-out validation articles, never from the fitting set, so its curve
is out-of-sample like the steered one.
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
    article_val_split, criterion_maps, mean_ci, split_by_outcome, wilson,
)
from judgeprobe.probes import (  # noqa: E402
    accuracy, best_band, fit_mass_mean, label_layers,
)
from judgeprobe.store import ActivationStore  # noqa: E402
from judgeprobe.transcripts import read_jsonl  # noqa: E402


def round_keys_of(store: ActivationStore, tid: str) -> list[str]:
    """The truncations captured for one debate, ordered r1, r2, ..., full."""
    names = {f.split("__")[0] for f in store.manifest["items"][tid]["files"]}
    rounds = sorted((n for n in names if n.startswith("r")),
                    key=lambda n: int(n[1:]))
    return rounds + (["full"] if "full" in names else [])


def order_mean(store: ActivationStore, tid: str, rk: str, cls: str) -> np.ndarray:
    """Mean activation across the two presentation orders, one round, one class."""
    return (store.load(tid, f"{rk}__forward__{cls}")
            + store.load(tid, f"{rk}__reverse__{cls}")) / 2.0


def stacked(store, ids, rk, layer):
    """`(pos, neg)` at one round and one layer, for a list of debates."""
    pos = np.stack([order_mean(store, i, rk, "gold")[layer] for i in ids])
    neg = np.stack([order_mean(store, i, rk, "distractor")[layer] for i in ids])
    return pos, neg


def curve(store, ids, rounds, layer, direction, chose_gold) -> list[dict]:
    """Probe margins and accuracy at each round, for one set of debates.

    Two sign conventions, because Phase 3 changed what the number means:

    * `margin_gold` -- toward the **gold** answer. Q2 as the doc poses it.
    * `margin_verdict` -- toward the answer the judge **actually gave**. Since the
      direction turned out to be the verdict direction, this is the one that is
      comparable between a steered set (judge chose the distractor) and an honest
      control (judge chose gold): both curves then measure the same thing, how
      strongly the residual stream has committed to the answer that will come out.
      Comparing raw `margin_gold` across the two sets compares mirror images and
      would read a shared length effect as an opposite-direction one.
    """
    out = []
    for rk in rounds:
        have = [i for i in ids if rk in round_keys_of(store, i)]
        if not have:
            continue
        pos, neg = stacked(store, have, rk, layer)
        m_gold = pos @ direction - neg @ direction
        sign = np.array([1.0 if chose_gold[i] else -1.0 for i in have])
        m_verdict = m_gold * sign
        k = int(np.sum(m_gold > 0))
        mg, glo, ghi = mean_ci(m_gold)
        mv, vlo, vhi = mean_ci(m_verdict)
        acc_lo, acc_hi = wilson(k, len(have))
        out.append({"round": rk, "n": len(have),
                    "acc": k / len(have), "acc_ci95": [acc_lo, acc_hi],
                    "mean_margin_gold": mg, "margin_gold_ci95": [glo, ghi],
                    "mean_margin_verdict": mv, "margin_verdict_ci95": [vlo, vhi],
                    "margins_gold": [float(x) for x in m_gold]})
    return out


def show(name: str, rows: list[dict]) -> None:
    print(f"\n{name}")
    print(f"  {'round':6s} {'n':>4s} {'P(gold)acc':>11s} {'margin->gold':>13s} "
          f"{'margin->verdict':>16s}  {'95% CI (verdict)':>22s}")
    for r in rows:
        print(f"  {r['round']:6s} {r['n']:4d} {r['acc']:11.3f} "
              f"{r['mean_margin_gold']:13.3f} {r['mean_margin_verdict']:16.3f}  "
              f"[{r['margin_verdict_ci95'][0]:9.3f}, "
              f"{r['margin_verdict_ci95'][1]:9.3f}]")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--acts", type=Path, default=config.PHASE2_DIR / "acts_rounds")
    ap.add_argument("--transcripts", type=Path,
                    default=config.PHASE2_DIR / "transcripts.jsonl")
    ap.add_argument("--criterion", choices=("verdict", "forced"), default="verdict")
    ap.add_argument("--layer", type=int, default=None,
                    help="pin the store-row layer index (default: select on the "
                         "held-out validation articles, as Phase 3 does)")
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
    rounds = sorted({rk for i in ids for rk in round_keys_of(store, i)},
                    key=lambda n: (n == "full", 0 if n == "full" else int(n[1:])))
    print(f"{len(ids)} debates, truncations captured: {rounds}")
    if len(rounds) < 2:
        print("only one capture point -- rerun the harness with --rounds 1,2,full "
              "into a FRESH store directory (resume skips by transcript id, not by "
              "round key, so reusing the full-only store writes nothing)")
        return 1
    if rounds == ["r1", "full"]:
        print("note: every debate is 2 rounds, so this is a two-point comparison "
              "(after round 1 vs after round 2), not a curve")

    decided, chose_gold = criterion_maps(ids, rows, args.criterion)
    steered, train, excluded, test_articles = split_by_outcome(
        ids, condition_of=condition_of, article_of=article_of,
        decided=decided, chose_gold=chose_gold)
    fit_ids, val_ids, _ = article_val_split(train, article_of,
                                            val_frac=args.val_frac, seed=args.seed)
    control = [i for i in val_ids if condition_of[i] == "honest"]
    print(f"\nsplit (criterion: {args.criterion}): "
          f"fit {len(fit_ids)}, val {len(val_ids)}, steered {len(steered)}, "
          f"control (held-out honest, judge right) {len(control)}")
    if len(steered) < 5 or len(fit_ids) < 20 or len(control) < 5:
        print("not enough data to fit; stopping")
        return 1

    # --- the probe, trained once, on full transcripts ------------------------
    P_fit = np.stack([order_mean(store, i, "full", "gold") for i in fit_ids])
    N_fit = np.stack([order_mean(store, i, "full", "distractor") for i in fit_ids])
    P_val = np.stack([order_mean(store, i, "full", "gold") for i in val_ids])
    N_val = np.stack([order_mean(store, i, "full", "distractor") for i in val_ids])

    if args.layer is None:
        sweep = []
        for layer in range(P_fit.shape[1]):
            pr = fit_mass_mean(P_fit[:, layer, :], N_fit[:, layer, :], layer=layer)
            sweep.append({"layer": layer,
                          "val_acc": accuracy(pr, P_val[:, layer, :],
                                              N_val[:, layer, :]),
                          "train_acc": accuracy(pr, P_fit[:, layer, :],
                                                N_fit[:, layer, :])})
        sweep = label_layers(sweep, store.manifest.get("layers"))
        band = best_band(sweep, key="val_acc")
        chosen = band["layer"] if band else max(sweep, key=lambda r: r["val_acc"])["layer"]
        chosen_row = next(r for r in sweep if r["layer"] == chosen)
        model_layer = chosen_row.get("model_layer", chosen)
        print(f"layer selected on validation articles: model layer {model_layer} "
              f"(val acc {chosen_row['val_acc']:.3f})")
    else:
        chosen = args.layer
        layers = store.manifest.get("layers") or []
        model_layer = layers[chosen] if chosen < len(layers) else chosen
        print(f"layer pinned: store row {chosen} = model layer {model_layer}")

    probe = fit_mass_mean(P_fit[:, chosen, :], N_fit[:, chosen, :], layer=chosen)

    # --- the two curves ------------------------------------------------------
    steered_curve = curve(store, steered, rounds, chosen, probe.direction, chose_gold)
    control_curve = curve(store, control, rounds, chosen, probe.direction, chose_gold)
    show("steered debates (judge wrong)", steered_curve)
    show("control: held-out honest debates the judge got right", control_curve)

    # --- reading the curves --------------------------------------------------
    print("\nreading -- commitment to the answer the judge gives (margin -> verdict)")
    deltas = {}
    for name, rows_ in (("steered", steered_curve), ("control", control_curve)):
        first, last = rows_[0], rows_[-1]
        delta = last["mean_margin_verdict"] - first["mean_margin_verdict"]
        deltas[name] = delta
        overlap = not (first["margin_verdict_ci95"][1] < last["margin_verdict_ci95"][0]
                       or last["margin_verdict_ci95"][1] < first["margin_verdict_ci95"][0])
        shape = "flat" if overlap else ("rising" if delta > 0 else "falling")
        print(f"  {name:8s}: {first['round']} -> {last['round']} "
              f"{first['mean_margin_verdict']:+.3f} -> "
              f"{last['mean_margin_verdict']:+.3f} ({delta:+.3f}, {shape})")
    print(f"  difference in movement (steered - control): "
          f"{deltas['steered'] - deltas['control']:+.3f}")
    print("  Both curves moving together is a context-length effect, which is what "
          "the control is for; only the difference between them is about the debate.")

    report = {
        "criterion": args.criterion,
        "layer": int(chosen), "model_layer": int(model_layer),
        "rounds": rounds,
        "n_fit": len(fit_ids), "n_val": len(val_ids),
        "n_steered": len(steered), "n_control": len(control),
        "n_excluded": len(excluded),
        "steered_curve": steered_curve,
        "control_curve": control_curve,
        "steered_ids": steered,
        "control_ids": control,
    }
    suffix = "" if args.criterion == "verdict" else f"_{args.criterion}"
    out = args.out or (args.acts / f"phase5_report{suffix}.json")
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nreport -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
