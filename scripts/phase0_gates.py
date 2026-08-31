"""Phase 0 gates A, B and C (design doc s0.2-s0.4).

    Gate A  the judge runs and produces parseable verdicts
    Gate B  activations cache with sane shapes
    Gate C  a probe separates true from false on easy examples

Do not proceed past Phase 0 until all three pass. If Gate C fails on easy data,
the doc says take the s6 fallback rather than scaling up.

Usage:
    python scripts/phase0_gates.py --model meta-llama/Llama-3.1-8B-Instruct
    python scripts/phase0_gates.py --model <id> --gate b      # one gate
    python scripts/phase0_gates.py --gate c                   # reuse cached acts
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from judgeprobe import config, probes  # noqa: E402
from judgeprobe.transcripts import read_jsonl  # noqa: E402

TRANSCRIPTS = config.PHASE0_DIR / "transcripts.jsonl"
VERDICTS = config.PHASE0_DIR / "verdicts.json"
ACTS = config.PHASE0_DIR / "activations.npz"
GATE_REPORT = config.PHASE0_DIR / "gate_report.json"

# Gate thresholds, fixed here so they are not chosen after seeing the numbers.
GATE_A_MIN_PARSE_RATE = 1.0   # all 10 must parse; a flaky format is fixed now, not later
GATE_C_MIN_LOO_ACC = 0.75     # best-layer leave-one-pair-out accuracy
GATE_C_MIN_GOOD_LAYERS = 3    # a band of working layers, not one lucky spike


def _load_transcripts():
    if not TRANSCRIPTS.exists():
        raise SystemExit(f"no transcripts at {TRANSCRIPTS} -- run "
                         "scripts/phase0_build_transcripts.py first")
    return read_jsonl(TRANSCRIPTS)


def _load_judge(model_id: str):
    from judgeprobe.judge import Judge

    print(f"loading judge {model_id} ...", flush=True)
    t0 = time.time()
    judge = Judge(model_id)
    print(f"  loaded in {time.time() - t0:.1f}s  device={judge.device} "
          f"dtype={judge.dtype} layers={judge.n_layers} d_model={judge.d_model}")
    return judge


# --- Gate A -------------------------------------------------------------------

def gate_a(judge, transcripts) -> dict:
    print("\n=== Gate A: parseable verdicts (counterbalanced) ===")
    print("  each debate judged twice, with the two answers trading slots")
    rows, cbs = [], []
    for t in transcripts:
        cb = judge.verdict_pair(t)
        cbs.append(cb)
        rows.append(cb.to_dict() | {"condition": t.condition})
        both_parsed = cb.forward.parsed and cb.reverse.parsed
        flag = "ok " if both_parsed else "FAIL"
        print(f"  [{flag}] {t.transcript_id[:38]:38s} "
              f"fwd={str(cb.forward.letter):4s} rev={str(cb.reverse.letter):4s} "
              f"{'AGREE ' if cb.consistent else 'ORDER!'} "
              f"choice={str(cb.choice):11s} p_gold={cb.p_gold:.3f}")

    VERDICTS.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    n = len(rows)

    # Gate criterion stays parseability, per the design doc -- but now over both
    # presentation orders, so 2n generations must all parse.
    n_parsed = sum(cb.forward.parsed + cb.reverse.parsed for cb in cbs)
    parse_rate = n_parsed / (2 * n)

    consistent = [cb for cb in cbs if cb.consistent]
    consistency_rate = len(consistent) / n
    acc = (sum(cb.correct for cb in consistent) / len(consistent)) if consistent else None

    # The headline position-bias number: across all 2n runs, how often did the
    # judge pick whichever answer was printed first? ~50% is a content-driven
    # judge; ~100% (or ~0%) means presentation order decided every verdict.
    picks = [v.chose_first_slot for cb in cbs for v in (cb.forward, cb.reverse)]
    picks = [p for p in picks if p is not None]
    first_slot_rate = sum(picks) / len(picks) if picks else float("nan")

    # Order-averaged, so this is the debiased view of Q0's mechanics on fake data.
    by_cond = {
        c: (sum(cb.forced_choice == "gold" for cb, r in zip(cbs, rows) if r["condition"] == c)
            / max(1, sum(r["condition"] == c for r in rows)))
        for c in ("clean", "misleading")
    }

    result = {
        "gate": "A",
        "n": n,
        "n_generations": 2 * n,
        "parse_rate": parse_rate,
        "order_consistency_rate": consistency_rate,
        "verdict_accuracy_consistent_only": acc,
        "forced_choice_gold_rate_by_condition": by_cond,
        "first_slot_rate": first_slot_rate,
        "passed": parse_rate >= GATE_A_MIN_PARSE_RATE,
    }
    print(f"  parse rate {parse_rate:.0%} over {2 * n} generations "
          f"(need {GATE_A_MIN_PARSE_RATE:.0%})")
    print(f"  order consistency {consistency_rate:.0%} -- the rest had their verdict "
          f"decided by presentation order")
    print(f"  verdict accuracy on order-consistent debates: "
          f"{'n/a' if acc is None else f'{acc:.0%}'}")
    print(f"  order-averaged gold rate: clean {by_cond['clean']:.0%}  "
          f"misleading {by_cond['misleading']:.0%}")
    print(f"  position bias: picked the first-listed option in {first_slot_rate:.0%} "
          f"of {len(picks)} runs (50% = content-driven)")
    if first_slot_rate >= 0.9 or first_slot_rate <= 0.1:
        print("  WARNING: the judge is answering by position, not by content.\n"
              "           Counterbalancing correctly voids these verdicts, but that "
              "leaves nothing to measure:\n"
              "           Q0 is undefined for this judge. Fix the prompt or move to a "
              "larger judge before Phase 1.")
    print(f"  Gate A: {'PASS' if result['passed'] else 'FAIL'}")
    return result


# --- Gate B -------------------------------------------------------------------

def gate_b(judge, transcripts) -> dict:
    print("\n=== Gate B: activation caching (both presentation orders) ===")
    acts = {("gold", "forward"): [], ("distractor", "forward"): [],
            ("gold", "reverse"): [], ("distractor", "reverse"): []}
    problems = []

    for i, t in enumerate(transcripts):
        # Verify the hooks against HF's own hidden_states on the first example.
        # If these disagree, every activation downstream is off by a layer.
        #
        # All but the last layer must match exactly. The last is expected to
        # differ: HF applies the final norm before appending the last entry of
        # `hidden_states`, so `hidden_states[-1] == model.norm(layer_{n-1}_out)`.
        # The hooks deliberately keep the raw residual stream at every layer --
        # probing a post-norm vector at the last layer and pre-norm everywhere
        # else would make the layer sweep incommensurable.
        if i == 0:
            p0 = judge.contrast_prompts(t)["gold"]
            a, hs = judge.residuals(p0, return_hidden_states=True)
            body_dev = float(np.abs(a[:-1] - hs[:-1]).max())
            normed = judge.model.model.norm(
                torch.tensor(a[-1], device=judge.model.device, dtype=judge.dtype)[None]
            ).float().detach().cpu().numpy()[0]
            final_dev = float(np.abs(normed - hs[-1]).max())
            print(f"  hook-vs-hidden_states: layers 0..n-2 max |diff| = {body_dev:.2e}, "
                  f"final layer after norm = {final_dev:.2e}")
            if body_dev > 1e-3:
                problems.append(f"hooks disagree with hidden_states ({body_dev:.2e})")
            if final_dev > 1e-1:
                problems.append(f"final layer is not norm(hook) ({final_dev:.2e}) -- "
                                "check where this architecture applies its final norm")

        pair = judge.contrast_activations_pair(t)
        for order in ("forward", "reverse"):
            for cls in ("gold", "distractor"):
                arr = pair[order][cls]
                if arr.shape != (judge.n_layers, judge.d_model):
                    problems.append(f"{t.transcript_id}/{order}/{cls}: shape {arr.shape}")
                if not np.isfinite(arr).all():
                    problems.append(f"{t.transcript_id}/{order}/{cls}: non-finite values")
                # "magnitudes are plausible" (design doc s0.3). Deliberately a wide
                # band -- residual norms grow with depth and vary by architecture,
                # so this catches a dead or exploded stream, not a subtle drift.
                norms = np.linalg.norm(arr, axis=1)
                if norms.min() < 1e-3 or norms.max() > 1e6:
                    problems.append(
                        f"{t.transcript_id}/{order}/{cls}: implausible residual norms "
                        f"[{norms.min():.3g}, {norms.max():.3g}]")
                acts[(cls, order)].append(arr)
            if np.allclose(pair[order]["gold"], pair[order]["distractor"]):
                problems.append(f"{t.transcript_id}/{order}: contrast pair is identical")

        g_f, d_f = pair["forward"]["gold"], pair["forward"]["distractor"]
        g_r = pair["reverse"]["gold"]
        cos = lambda x, y: float(x @ y / (np.linalg.norm(x) * np.linalg.norm(y)))
        print(f"  [{i + 1:2d}/{len(transcripts)}] {t.transcript_id[:34]:34s} "
              f"tok={judge.n_prompt_tokens(judge.contrast_prompts(t)['gold']):5d} "
              f"|gold|={np.linalg.norm(g_f[-1]):7.1f} "
              f"cos(gold,dist)={cos(g_f[-1], d_f[-1]):.4f} "
              f"cos(fwd,rev)={cos(g_f[-1], g_r[-1]):.4f}")

    stacked = {f"{cls}_{order}": np.stack(v) for (cls, order), v in acts.items()}
    # Pre-registered representation for Phase 3: the mean over presentation orders,
    # which is order-invariant by construction. The single-order arrays are kept so
    # the effect of counterbalancing on the probe can be reported.
    stacked["gold"] = (stacked["gold_forward"] + stacked["gold_reverse"]) / 2
    stacked["distractor"] = (stacked["distractor_forward"] + stacked["distractor_reverse"]) / 2

    np.savez_compressed(
        ACTS,
        transcript_ids=np.array([t.transcript_id for t in transcripts]),
        conditions=np.array([t.condition for t in transcripts]),
        model_id=np.array(judge.model_id),
        **stacked,
    )
    result = {
        "gate": "B",
        "shape": list(stacked["gold"].shape),
        "n_layers": judge.n_layers,
        "d_model": judge.d_model,
        "orders_cached": ["forward", "reverse", "mean"],
        "problems": problems,
        "passed": not problems,
        "path": str(ACTS),
    }
    print(f"  cached {stacked['gold'].shape} x2 orders (+ their mean) -> {ACTS}")
    for p in problems:
        print(f"  PROBLEM: {p}")
    print(f"  Gate B: {'PASS' if result['passed'] else 'FAIL'}")
    return result


# --- Gate C -------------------------------------------------------------------

def gate_c() -> dict:
    print("\n=== Gate C: probe separates true from false ===")
    if not ACTS.exists():
        raise SystemExit(f"no activations at {ACTS} -- run Gate B first")
    z = np.load(ACTS, allow_pickle=False)
    gold, distractor = z["gold"], z["distractor"]  # order-averaged (see Gate B)
    conditions = z["conditions"]
    print(f"  using order-averaged activations, {gold.shape[0]} pairs")

    rows = probes.sweep_layers(gold, distractor, fit=probes.fit_mass_mean, loo=True)
    best = probes.best_layer(rows, key="loo_acc")
    good = [r for r in rows if r["loo_acc"] >= GATE_C_MIN_LOO_ACC]

    print(f"  {'layer':>5}  {'train':>6}  {'loo':>6}   (every 4th layer shown)")
    for r in rows:
        if r["layer"] % 4 == 0 or r["layer"] == best["layer"]:
            mark = " <- best" if r["layer"] == best["layer"] else ""
            print(f"  {r['layer']:5d}  {r['train_acc']:6.2f}  {r['loo_acc']:6.2f}{mark}")

    # The doc asks specifically about the clean examples: fit on everything else,
    # score the held-out clean pairs.
    clean_idx = [i for i, c in enumerate(conditions) if c == "clean"]
    clean_correct = 0
    L = best["layer"]
    for i in clean_idx:
        keep = [j for j in range(len(gold)) if j != i]
        p = probes.fit_mass_mean(gold[keep, L, :], distractor[keep, L, :], layer=L)
        clean_correct += int(p.predict(gold[i, L, :]).item() == 1)
        clean_correct += int(p.predict(distractor[i, L, :]).item() == 0)
    clean_acc = clean_correct / (2 * len(clean_idx)) if clean_idx else None

    # Secondary comparison only (design doc s0.5), so it is fitted at the layer
    # mass-mean selected rather than swept -- a full LBFGS sweep with leave-one-out
    # is ~30x the cost of the primary method for a number nothing depends on.
    log_loo = probes.loo_accuracy(gold[:, L, :], distractor[:, L, :],
                                  fit=probes.fit_logistic)
    log_probe = probes.fit_logistic(gold[:, L, :], distractor[:, L, :], layer=L)
    mm_probe = probes.fit_mass_mean(gold[:, L, :], distractor[:, L, :], layer=L)
    log_cos = probes.cosine(mm_probe, log_probe)

    # Does counterbalancing help the probe, or only the verdict? Cheap to answer,
    # and the answer belongs in the writeup either way.
    single_order = {}
    for order in ("forward", "reverse"):
        if f"gold_{order}" not in z:
            continue
        r = probes.sweep_layers(z[f"gold_{order}"], z[f"distractor_{order}"], loo=True)
        single_order[order] = float(probes.best_layer(r, key="loo_acc")["loo_acc"])
    if single_order:
        print("  best loo by presentation order: "
              + "  ".join(f"{k}={v:.2f}" for k, v in single_order.items())
              + f"  averaged={best['loo_acc']:.2f}")

    result = {
        "gate": "C",
        "n_pairs": int(len(gold)),
        "single_order_best_loo_acc": single_order,
        "best_layer": int(best["layer"]),
        "best_loo_acc": float(best["loo_acc"]),
        "best_train_acc": float(best["train_acc"]),
        "clean_only_loo_acc": clean_acc,
        "n_layers_above_threshold": len(good),
        "logistic_loo_acc_at_best_layer": float(log_loo),
        "logistic_vs_mass_mean_cosine": float(log_cos),
        "layer_sweep": rows,
        "passed": (best["loo_acc"] >= GATE_C_MIN_LOO_ACC
                   and len(good) >= GATE_C_MIN_GOOD_LAYERS),
    }
    print(f"  best layer {best['layer']}: loo {best['loo_acc']:.2f} "
          f"(need >= {GATE_C_MIN_LOO_ACC}), {len(good)}/{len(rows)} layers above "
          f"threshold (need >= {GATE_C_MIN_GOOD_LAYERS})")
    print(f"  clean-only held-out accuracy at layer {L}: "
          f"{'n/a' if clean_acc is None else f'{clean_acc:.2f}'}")
    print(f"  logistic (secondary) at layer {L}: loo {log_loo:.2f}, "
          f"cosine with mass-mean {log_cos:+.3f}")
    print("  NOTE: best layer is selected over all layers on 10 pairs -- treat as a "
          "plumbing check, not evidence. Layer selection moves to a validation "
          "split in Phase 3.")
    print(f"  Gate C: {'PASS' if result['passed'] else 'FAIL'}")
    return result


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=config.JUDGE_MODEL_PHASE0)
    ap.add_argument("--gate", choices=["a", "b", "c", "all"], default="all")
    args = ap.parse_args()

    config.ensure_dirs()
    results = {}
    needs_model = args.gate in ("a", "b", "all")

    if needs_model:
        transcripts = _load_transcripts()
        judge = _load_judge(args.model)
        if args.gate in ("a", "all"):
            results["A"] = gate_a(judge, transcripts)
        if args.gate in ("b", "all"):
            results["B"] = gate_b(judge, transcripts)
    if args.gate in ("c", "all"):
        results["C"] = gate_c()

    report = {
        "model": args.model,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "gates": results,
        "all_passed": all(r["passed"] for r in results.values()),
    }
    GATE_REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\n" + "=" * 60)
    for name, r in results.items():
        print(f"  Gate {name}: {'PASS' if r['passed'] else 'FAIL'}")
    print(f"  report -> {GATE_REPORT}")
    if not report["all_passed"]:
        print("\n  STOP. Do not start Phase 1. Record the failure in logbook.md;\n"
              "  if Gate C failed on easy data, take the s6 fallback rather than scaling up.")
    print("=" * 60)
    return 0 if report["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
