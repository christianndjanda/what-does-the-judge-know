"""Is low order-consistency the judge, or the stimuli?

Gate A on the Phase 0 corpus gave order consistency 0.10 -- nine of ten debates
gave opposite answers depending only on slot order. That is ambiguous:

  (a) the judge is position-anchored on this prompt format, or
  (b) the templated Phase 0 transcripts carry no discriminating signal, so the
      judge has nothing to go on and falls back to position.

(a) means fix the prompt or the judge before Phase 1. (b) means the number says
nothing yet and should be re-measured on real debates in Phase 2. Opposite actions,
so it is worth five minutes to separate them.

The test: give the judge items where the right answer is not in doubt, and see
whether consistency recovers.

  tier 1  world knowledge, no debate at all -- the cleanest ceiling. If the judge
          cannot answer "capital of France" consistently under a slot swap, the
          prompt format is broken and nothing else matters.
  tier 2  the same items with a one-line debate, one side plainly right. Isolates
          whether the debate framing itself induces the anchoring.
  tier 3  real Phase 0 debates, but with the passage excerpt shown to the judge
          (`include_article=True`). Note the excerpt is the first N characters of
          the article and may not contain the answer, so a low score here is weaker
          evidence than tiers 1-2.

Usage:
    python scripts/diagnose_judge_ceiling.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from judgeprobe import config  # noqa: E402
from judgeprobe.judge import Judge  # noqa: E402
from judgeprobe.transcripts import Transcript, Turn, read_jsonl  # noqa: E402

# (question, correct, wrong, one-line argument for the correct side)
EASY = [
    ("What is the capital of France?", "Paris", "Berlin",
     "Paris has been the capital of France for centuries; Berlin is in Germany."),
    ("How many legs does a spider have?", "Eight", "Six",
     "Spiders are arachnids, which have eight legs. Six-legged creatures are insects."),
    ("Which planet is closest to the Sun?", "Mercury", "Neptune",
     "Mercury orbits nearest the Sun; Neptune is the outermost planet."),
    ("What is the largest ocean on Earth?", "The Pacific Ocean", "The Arctic Ocean",
     "The Pacific covers about a third of the planet; the Arctic is the smallest ocean."),
    ("Water freezes at what temperature at sea level?", "0 degrees Celsius",
     "100 degrees Celsius",
     "Water freezes at 0C and boils at 100C, so 100 is the boiling point, not freezing."),
    ("Who wrote the play Hamlet?", "William Shakespeare", "Charles Dickens",
     "Hamlet is Shakespeare's; Dickens was a novelist writing two centuries later."),
]


def easy_transcript(i, question, correct, wrong, argument, *, with_debate: bool):
    turns = []
    if with_debate:
        turns = [
            Turn(1, "gold", "Debater A", argument),
            Turn(1, "distractor", "Debater B",
                 f"I say {wrong.rstrip('.')}. I have no particular support for this."),
        ]
    return Transcript(
        transcript_id=f"ceiling-{'debate' if with_debate else 'bare'}-{i}",
        question_id=f"ceiling-{i}",
        question=question,
        gold=correct,
        best_distractor=wrong,
        condition="ceiling",
        turns=turns,
        # Balanced by hand: alternate, so the tier is not confounded by layout.
        gold_letter="A" if i % 2 == 0 else "B",
        meta={"synthetic": True, "tier": "debate" if with_debate else "bare"},
    )


def run_tier(judge, transcripts, label, **kw):
    print(f"\n=== {label} ===")
    cbs = []
    for t in transcripts:
        cb = judge.verdict_pair(t, **kw)
        cbs.append(cb)
        print(f"  {t.transcript_id[:34]:34s} fwd={str(cb.forward.letter):4s} "
              f"rev={str(cb.reverse.letter):4s} "
              f"{'AGREE ' if cb.consistent else 'ORDER!'} "
              f"choice={str(cb.choice):11s} p_gold={cb.p_gold:.3f}", flush=True)

    n = len(cbs)
    consistency = sum(cb.consistent for cb in cbs) / n
    consistent = [cb for cb in cbs if cb.consistent]
    acc = sum(cb.correct for cb in consistent) / len(consistent) if consistent else None
    picks = [v.chose_first_slot for cb in cbs for v in (cb.forward, cb.reverse)]
    picks = [p for p in picks if p is not None]
    first_slot = sum(picks) / len(picks) if picks else float("nan")

    print(f"  order consistency {consistency:.0%}   "
          f"accuracy (consistent only) {'n/a' if acc is None else f'{acc:.0%}'}   "
          f"first-slot {first_slot:.0%}")
    return {"label": label, "n": n, "order_consistency": consistency,
            "accuracy_consistent_only": acc, "first_slot_rate": first_slot}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=config.JUDGE_MODEL_PHASE0)
    args = ap.parse_args()

    judge = Judge(args.model)
    results = []

    results.append(run_tier(
        judge,
        [easy_transcript(i, *q, with_debate=False) for i, q in enumerate(EASY)],
        "tier 1: world knowledge, no debate"))

    results.append(run_tier(
        judge,
        [easy_transcript(i, *q, with_debate=True) for i, q in enumerate(EASY)],
        "tier 2: world knowledge, one-line debate"))

    path = config.PHASE0_DIR / "transcripts.jsonl"
    if path.exists():
        results.append(run_tier(
            judge, read_jsonl(path),
            "tier 3: Phase 0 debates, passage excerpt shown to judge",
            include_article=True))

    print("\n" + "=" * 66)
    for r in results:
        print(f"  {r['label']:52s} {r['order_consistency']:.0%}")
    print("=" * 66)

    t1 = results[0]["order_consistency"]
    print("\n  reading:")
    if t1 >= 0.8:
        print("    Tier 1 is high, so the prompt format and the judge are fine. The 0.10")
        print("    on the Phase 0 corpus is about the STIMULI -- templated boilerplate")
        print("    with nothing to separate. Do not change the judge or the prompt on")
        print("    the strength of that number; re-measure on real debates in Phase 2.")
    elif t1 <= 0.4:
        print("    Tier 1 is low, so the judge cannot answer even trivial questions")
        print("    consistently under a slot swap. This is the PROMPT FORMAT or the")
        print("    judge, not the stimuli, and it must be fixed before Phase 1 --")
        print("    Q0 cannot be measured with a judge in this state.")
    else:
        print("    Tier 1 is middling: some position anchoring survives even on trivial")
        print("    items. Worth one prompt iteration, but the Phase 0 corpus is likely")
        print("    contributing as well. Compare tiers 1 and 2 to see whether the debate")
        print("    framing itself is what induces the anchoring.")

    out = config.PHASE0_DIR / "judge_ceiling.json"
    out.write_text(json.dumps({"model": args.model, "tiers": results}, indent=2),
                   encoding="utf-8")
    print(f"\n  wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
