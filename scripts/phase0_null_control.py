"""Null control for Gate C: does the probe need the debate at all?

Gate C fits a probe on contrast pairs that differ by the appended answer text:

    [question][options][transcript] The answer is {gold}.
    [question][options][transcript] The answer is {best_distractor}.

A probe can separate those without ever reading the transcript -- for instance if
QuALITY gold answers are systematically longer, more specific, or more plausible
in isolation than their distractors. That would score highly on Gate C while
measuring a dataset artifact rather than anything the judge concluded.

This script rebuilds the identical contrast pairs with the debate removed and
refits. Reading:

  null LOO ~ chance          the probe needs the debate. Gate C means what it says.
  null LOO ~ Gate C's LOO    the probe is reading the answer strings, not the
                             judge's assessment. Gate C is not evidence, and the
                             contrast-pair design needs rethinking before Phase 3.

Also reports the cosine between the null direction and the Gate C direction at the
same layer: high cosine is the same warning in a different form.

Usage:
    python scripts/phase0_null_control.py            # after gates B and C
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from judgeprobe import config, probes  # noqa: E402
from judgeprobe.judge import (  # noqa: E402
    SYSTEM_PROMPT, VERDICT_INSTRUCTION, Judge,
)
from judgeprobe.transcripts import read_jsonl  # noqa: E402

NULL_DEBATE = "(no debate took place)"


def null_contrast_prompts(judge: Judge, transcript, *, swap: bool = False) -> dict[str, str]:
    """The Gate B contrast pairs, with the transcript replaced by a placeholder."""
    layout = judge.layout(transcript, swap=swap)
    body = [
        f"Question: {transcript.question}",
        f"A. {layout.option_a}",
        f"B. {layout.option_b}",
        "\nDebate transcript:",
        NULL_DEBATE,
        "\n" + VERDICT_INSTRUCTION,
    ]
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "\n".join(body)},
    ]
    out = {}
    for cls, answer in (("gold", transcript.gold),
                        ("distractor", transcript.best_distractor)):
        out[cls] = judge._chat(messages, add_generation_prompt=True,
                               continuation=f"The answer is {answer.rstrip('.')}.")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=config.JUDGE_MODEL_PHASE0)
    args = ap.parse_args()

    transcripts = read_jsonl(config.PHASE0_DIR / "transcripts.jsonl")
    acts_path = config.PHASE0_DIR / "activations.npz"
    if not acts_path.exists():
        raise SystemExit(f"no Gate B activations at {acts_path} -- run the gates first")
    real = np.load(acts_path, allow_pickle=False)

    judge = Judge(args.model)
    print(f"building null contrast pairs for {len(transcripts)} transcripts "
          f"(both presentation orders)")

    gold, distractor = [], []
    for i, t in enumerate(transcripts):
        per_order = {"gold": [], "distractor": []}
        for swap in (False, True):
            for cls, prompt in null_contrast_prompts(judge, t, swap=swap).items():
                per_order[cls].append(judge.residuals(prompt))
        # Same order-averaging as Gate B, so the comparison is like for like.
        gold.append(np.mean(per_order["gold"], axis=0))
        distractor.append(np.mean(per_order["distractor"], axis=0))
        print(f"  [{i + 1:2d}/{len(transcripts)}] {t.transcript_id[:44]}", flush=True)

    gold_arr, distractor_arr = np.stack(gold), np.stack(distractor)

    null_rows = probes.sweep_layers(gold_arr, distractor_arr, loo=True)
    null_best = probes.best_layer(null_rows, key="loo_acc")

    real_rows = probes.sweep_layers(real["gold"], real["distractor"], loo=True)
    real_best = probes.best_layer(real_rows, key="loo_acc")

    # Compare directions at the layer Gate C selected.
    L = real_best["layer"]
    real_probe = probes.fit_mass_mean(real["gold"][:, L, :], real["distractor"][:, L, :], layer=L)
    null_probe = probes.fit_mass_mean(gold_arr[:, L, :], distractor_arr[:, L, :], layer=L)
    cos = probes.cosine(real_probe, null_probe)

    # Does the *real* probe still work when the debate is gone?
    transfer = probes.accuracy(real_probe, gold_arr[:, L, :], distractor_arr[:, L, :])

    print(f"\n  with debate   : best layer {real_best['layer']}, loo {real_best['loo_acc']:.2f}")
    print(f"  without debate: best layer {null_best['layer']}, loo {null_best['loo_acc']:.2f}")
    print(f"  cosine(with, without) at layer {L}: {cos:+.3f}")
    print(f"  Gate C probe applied to null activations: {transfer:.2f}")

    print("\n  reading:")
    if real_best["loo_acc"] < 0.75:
        # Both arms at chance is not a failed control -- there was no signal to
        # control for. Reporting FAIL would blame the contrast-pair design for a
        # probe that never worked, a different problem with a different fix.
        print("    UNINFORMATIVE. The with-debate probe is itself near chance, so")
        print("    there is no Gate C result for this control to qualify. Fix Gate")
        print("    C first; this script means something only once the probe works.")
    elif null_best["loo_acc"] >= real_best["loo_acc"] - 0.1:
        print("    FAIL. The probe separates the classes just as well with no debate")
        print("    present, so it is reading the answer strings rather than anything")
        print("    the judge concluded. Gate C is not evidence. Rethink the contrast")
        print("    pairs before Phase 3.")
    elif null_best["loo_acc"] <= 0.65:
        print("    PASS. The probe needs the debate; without it separation collapses")
        print("    toward chance. Gate C measures something the judge computed.")
    else:
        print("    PARTIAL. Some separation survives without the debate, so Gate C's")
        print("    number is inflated by an answer-string component. Report both.")

    out = config.PHASE0_DIR / "null_control.json"
    out.write_text(json.dumps({
        "model": args.model,
        "with_debate": {"best_layer": int(real_best["layer"]),
                        "best_loo_acc": float(real_best["loo_acc"])},
        "without_debate": {"best_layer": int(null_best["layer"]),
                           "best_loo_acc": float(null_best["loo_acc"])},
        "cosine_at_gate_c_layer": float(cos),
        "gate_c_probe_on_null": float(transfer),
        "null_layer_sweep": null_rows,
    }, indent=2), encoding="utf-8")
    print(f"\n  wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
