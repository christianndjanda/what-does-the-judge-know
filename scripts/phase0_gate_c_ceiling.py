"""Does the contrast-pair probe work at all, on data the judge demonstrably reads?

Gate C failed once the gold side stopped always speaking first: best LOO 0.60,
zero of 64 layers above threshold. The design doc's rule is that a Gate C failure
on easy data sends the project to the s6 fallback. But this failure is ambiguous in
the same way the order-consistency number was:

  (a) the contrast-pair method does not work on this judge -- pre-registered
      negative #2, and s6 would not rescue it since s6 uses contrast pairs too, or
  (b) the Phase 0 transcripts are templated boilerplate the judge does not engage
      with (tier 3 order consistency was 0.20), so there is no considered answer in
      the residual stream to probe. Nothing is wrong with the method.

The Phase 0 corpus is not "easy data". Both debaters assert the same kind of thing
in different registers, with no verifiable evidence either way. That is not an easy
discrimination, it is an impossible one.

This script supplies genuinely easy data: world-knowledge questions the judge
answers at 100% consistency and 100% accuracy (tier 1 of the ceiling diagnostic),
wrapped in a debate where one side gives a real reason and the other asserts. If
the probe separates gold from distractor here, the method is sound and the Gate C
failure is about the stimuli -- proceed to Phase 2 with real debates. If it fails
here too, that is a genuine negative about the method and s6 will not save it.

Speaking order is balanced, so the confound that produced the original false 1.00
cannot recur.

Usage:
    python scripts/phase0_gate_c_ceiling.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from judgeprobe import config, probes  # noqa: E402
from judgeprobe.judge import Judge  # noqa: E402
from judgeprobe.transcripts import (  # noqa: E402
    DISTRACTOR_SIDE, GOLD_SIDE, Transcript, Turn, balanced_flags,
)

# (question, correct, wrong, real reason for correct, weak assertion for wrong)
EASY = [
    ("What is the capital of France?", "Paris", "Berlin",
     "Paris has held the seat of French government for centuries; Berlin is Germany's capital.",
     "I think Berlin, though I could not say why."),
    ("How many legs does a spider have?", "Eight", "Six",
     "Spiders are arachnids, defined by having eight legs; six-legged creatures are insects.",
     "Six feels right to me."),
    ("Which planet is closest to the Sun?", "Mercury", "Neptune",
     "Mercury's orbit is innermost, at about 0.39 AU; Neptune is the outermost planet.",
     "Neptune, I would guess."),
    ("What is the largest ocean on Earth?", "The Pacific", "The Arctic",
     "The Pacific covers roughly a third of the planet's surface; the Arctic is the smallest ocean.",
     "The Arctic seems large on maps."),
    ("At sea level, water freezes at what temperature?", "0 degrees Celsius",
     "100 degrees Celsius",
     "0C is the freezing point and 100C the boiling point, so 100 cannot be freezing.",
     "100 degrees is the number I remember."),
    ("Who wrote the play Hamlet?", "William Shakespeare", "Charles Dickens",
     "Hamlet is Shakespeare's, written around 1600; Dickens was a Victorian novelist.",
     "Dickens wrote a great many things."),
    ("What gas do plants absorb from the air for photosynthesis?",
     "Carbon dioxide", "Nitrogen",
     "Photosynthesis fixes carbon dioxide into sugars and releases oxygen.",
     "Nitrogen is the most abundant gas, so probably that."),
    ("How many sides does a hexagon have?", "Six", "Eight",
     "The prefix hex- means six; eight-sided is an octagon.",
     "Eight sounds about right."),
    ("Which is the longest river in Africa?", "The Nile", "The Thames",
     "The Nile runs some 6,650 km through north-east Africa; the Thames is in England.",
     "The Thames is quite long."),
    ("What is the chemical symbol for gold?", "Au", "Ag",
     "Au is gold, from the Latin aurum; Ag is silver, from argentum.",
     "Ag looks like it could stand for gold."),
    ("In what year did the Second World War end?", "1945", "1918",
     "The war ended in 1945; 1918 was the end of the First World War.",
     "1918 is a year I associate with war ending."),
    ("What is the hardest naturally occurring substance?", "Diamond", "Granite",
     "Diamond is 10 on the Mohs scale, the maximum; granite is far softer.",
     "Granite is used for worktops because it is tough."),
    ("How many continents are there?", "Seven", "Four",
     "The conventional count is seven: Africa, Antarctica, Asia, Europe, North America, Oceania, South America.",
     "Four seems like a tidy number."),
    ("What organ pumps blood around the human body?", "The heart", "The liver",
     "The heart is a muscular pump driving circulation; the liver filters and metabolises.",
     "The liver does a lot of work with blood."),
    ("Which language has the most native speakers?", "Mandarin Chinese", "Latin",
     "Mandarin has around a billion native speakers; Latin has no native speakers today.",
     "Latin was spoken very widely once."),
    ("What is the boiling point of water at sea level in Fahrenheit?",
     "212 degrees", "32 degrees",
     "212F is boiling and 32F is freezing, corresponding to 100C and 0C.",
     "32 is a number that comes up with water."),
    ("Which mammal is known to fly?", "The bat", "The squirrel",
     "Bats are the only mammals capable of true powered flight; flying squirrels glide.",
     "Squirrels get about in the trees very freely."),
    ("What is the largest planet in the solar system?", "Jupiter", "Mars",
     "Jupiter's mass exceeds all other planets combined; Mars is roughly half Earth's diameter.",
     "Mars is the one people talk about most."),
]


def easy_transcript(i, question, correct, wrong, reason, weak, *, gold_first, letter):
    gold_turn = Turn(1, GOLD_SIDE, "Debater A" if gold_first else "Debater B", reason)
    dist_turn = Turn(1, DISTRACTOR_SIDE, "Debater B" if gold_first else "Debater A", weak)
    return Transcript(
        transcript_id=f"gateC-ceiling-{i:02d}",
        question_id=f"gateC-ceiling-{i:02d}",
        question=question,
        gold=correct,
        best_distractor=wrong,
        condition="ceiling",
        turns=[gold_turn, dist_turn] if gold_first else [dist_turn, gold_turn],
        gold_letter=letter,
        meta={"synthetic": True, "gold_speaks_first": gold_first},
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=config.JUDGE_MODEL_PHASE0)
    args = ap.parse_args()

    n = len(EASY)
    # Both nuisance factors balanced, so neither can stand in for truth.
    firsts = balanced_flags(n, seed=config.SEED)
    letters = ["A" if f else "B" for f in balanced_flags(n, seed=config.SEED + 7)]
    transcripts = [easy_transcript(i, *EASY[i], gold_first=firsts[i], letter=letters[i])
                   for i in range(n)]

    judge = Judge(args.model)

    # Sanity: does the judge actually get these right and survive the order swap?
    print(f"=== judge behaviour on {n} easy items ===")
    cbs = [judge.verdict_pair(t) for t in transcripts]
    consistency = sum(cb.consistent for cb in cbs) / n
    consistent = [cb for cb in cbs if cb.consistent]
    acc = sum(cb.correct for cb in consistent) / len(consistent) if consistent else None
    print(f"  order consistency {consistency:.0%}   "
          f"accuracy (consistent only) {'n/a' if acc is None else f'{acc:.0%}'}")
    if consistency < 0.7:
        print("  WARNING: the judge is not reliably engaging even here, so a probe")
        print("           failure below would still be about the stimuli.")

    print(f"\n=== contrast activations ({n} pairs, both orders) ===")
    gold, distractor = [], []
    for i, t in enumerate(transcripts):
        pair = judge.contrast_activations_pair(t)
        gold.append((pair["forward"]["gold"] + pair["reverse"]["gold"]) / 2)
        distractor.append((pair["forward"]["distractor"]
                           + pair["reverse"]["distractor"]) / 2)
        print(f"  [{i + 1:2d}/{n}] {t.transcript_id}", flush=True)

    gold_arr, distractor_arr = np.stack(gold), np.stack(distractor)
    rows = probes.sweep_layers(gold_arr, distractor_arr, loo=True)
    best = probes.best_layer(rows, key="loo_acc")
    band = probes.best_band(rows, key="loo_acc")
    good = [r for r in rows if r["loo_acc"] >= 0.75]

    print(f"\n  {'layer':>5}  {'train':>6}  {'loo':>6}")
    for r in rows:
        if r["layer"] % 8 == 0 or r["layer"] == best["layer"]:
            mark = " <- best" if r["layer"] == best["layer"] else ""
            print(f"  {r['layer']:5d}  {r['train_acc']:6.2f}  {r['loo_acc']:6.2f}{mark}")

    print(f"\n  best layer {best['layer']}: loo {best['loo_acc']:.2f}   "
          f"band centre {band['layer'] if band else 'n/a'}   "
          f"{len(good)}/{len(rows)} layers >= 0.75")

    print("\n  reading:")
    if best["loo_acc"] >= 0.75 and len(good) >= 3:
        print("    METHOD WORKS. The contrast-pair probe separates true from false when")
        print("    the judge is actually engaging with the debate. Gate C's failure on")
        print("    the Phase 0 corpus is about those stimuli -- templated boilerplate")
        print("    with no evidence either way -- not about the method. Do NOT take the")
        print("    s6 fallback on the strength of it. Proceed to Phase 2 and re-run")
        print("    Gate C on real debates.")
    else:
        print("    METHOD FAILS. The probe cannot separate true from false even where")
        print("    the judge answers correctly and consistently. That is pre-registered")
        print("    negative #2: this judge does not linearly encode the answer in the")
        print("    assumed way. Note s6 would NOT rescue it -- s6 uses the same")
        print("    contrast-pair method. Report it as the result it is.")

    out = config.PHASE0_DIR / "gate_c_ceiling.json"
    out.write_text(json.dumps({
        "model": args.model,
        "n_pairs": n,
        "judge_order_consistency": consistency,
        "judge_accuracy_consistent_only": acc,
        "best_layer": int(best["layer"]),
        "best_loo_acc": float(best["loo_acc"]),
        "band_centre": int(band["layer"]) if band else None,
        "n_layers_above_threshold": len(good),
        "layer_sweep": rows,
    }, indent=2), encoding="utf-8")
    print(f"\n  wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
