"""Where does a judge's slot preference come from?

Not a gate -- a diagnostic to run once per judge model, before Phase 1. It
separates explanations that imply different fixes:

  normal     baseline P(first-listed option)
  swapped    the two answers trade slots. If this matches `normal`, the
             preference is attached to the position and not to the option text.
  numeric    labels A./B. -> 1./2.
  XY         labels A./B. -> X./Y.   If these match `normal`, the preference is
             not a lexical prior on the token "A"; relabelling will not fix it.
  no_debate  transcript removed. The residual preference here is the structural
             prior; the gap to `normal` is how much the debate text amplifies it.
             This number is also the contextual-calibration prior.

On SmolLM2-135M (Phase 0 smoke run): normal 0.805, swapped 0.811, numeric 0.807,
XY 0.861, no_debate 0.673 -- i.e. primacy over the first slot, content-independent,
and immune to relabelling. That is what motivated counterbalancing.

Usage:
    python scripts/diagnose_position_bias.py --model <hf-id> [-n 6]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from judgeprobe import config  # noqa: E402
from judgeprobe.judge import (  # noqa: E402
    SYSTEM_PROMPT, VERDICT_INSTRUCTION, Judge, build_judge_messages,
)
from judgeprobe.transcripts import read_jsonl  # noqa: E402

VARIANTS = ("normal", "swapped", "numeric", "XY", "no_debate")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=config.JUDGE_MODEL_PHASE0)
    ap.add_argument("-n", type=int, default=6, help="transcripts to probe")
    args = ap.parse_args()

    path = config.PHASE0_DIR / "transcripts.jsonl"
    if not path.exists():
        raise SystemExit(f"no transcripts at {path} -- build them first")
    transcripts = read_jsonl(path)[: args.n]
    judge = Judge(args.model)

    def p_first(messages, labels=("A", "B")) -> float:
        """Renormalised P(first label) at the answer position."""
        text = judge.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        enc = judge.tokenizer(text, return_tensors="pt", add_special_tokens=False)
        enc = {k: v.to(judge.model.device) for k, v in enc.items()}
        with torch.no_grad():
            logits = judge.model(**enc).logits[0, -1].float()
        ids = [judge._letter_token_id(l) for l in labels]
        return float(torch.softmax(logits[ids], dim=-1)[0])

    def relabelled(t, labels, opt1, opt2, transcript_text) -> list[dict]:
        instruction = VERDICT_INSTRUCTION.replace("A or B", f"{labels[0]} or {labels[1]}")
        body = [f"Question: {t.question}", f"{labels[0]}. {opt1}", f"{labels[1]}. {opt2}",
                "\nDebate transcript:", transcript_text, "\n" + instruction]
        return [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": "\n".join(body)}]

    rows = []
    for t in transcripts:
        layout = judge.layout(t)
        a, b = layout.option_a, layout.option_b
        row = {
            "normal": p_first(build_judge_messages(t, layout)),
            "swapped": p_first(build_judge_messages(t, layout.swapped())),
            "numeric": p_first(relabelled(t, ("1", "2"), a, b, t.render()),
                               labels=("1", "2")),
            "XY": p_first(relabelled(t, ("X", "Y"), a, b, t.render()), labels=("X", "Y")),
            "no_debate": p_first(relabelled(t, ("A", "B"), a, b, "(no debate took place)")),
        }
        rows.append(row | {"transcript_id": t.transcript_id})
        print(f"  {t.transcript_id[:32]:32s} "
              + "  ".join(f"{k}={row[k]:.3f}" for k in VARIANTS), flush=True)

    means = {k: float(np.mean([r[k] for r in rows])) for k in VARIANTS}
    print("\n  mean P(first-listed option):")
    for k in VARIANTS:
        print(f"    {k:10s} {means[k]:.3f}")

    print("\n  reading:")
    print(f"    content sensitivity : |normal - swapped| = "
          f"{abs(means['normal'] - means['swapped']):.3f}  (near 0 => the preference "
          f"is for the slot, not the answer)")
    print(f"    label dependence    : |normal - numeric| = "
          f"{abs(means['normal'] - means['numeric']):.3f}  (near 0 => relabelling "
          f"will not fix it)")
    print(f"    structural prior    : no_debate = {means['no_debate']:.3f}  "
          f"(0.5 = unbiased; this is the contextual-calibration prior)")

    out = config.PHASE0_DIR / "position_bias.json"
    out.write_text(json.dumps({"model": args.model, "means": means, "rows": rows},
                              indent=2), encoding="utf-8")
    print(f"\n  wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
