"""Phase 0.1 -- ten throwaway transcripts.

Five with a misleading argument for the best distractor, five clean. Templated
and deliberately artificial (design doc: "these are plumbing tests, not data"),
so this costs no API calls and reruns identically.

Usage:
    python scripts/phase0_build_transcripts.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from judgeprobe import config  # noqa: E402
from judgeprobe.quality import sample_questions  # noqa: E402
from judgeprobe.transcripts import (  # noqa: E402
    assign_balanced_letters, balanced_flags, letter_balance, make_phase0_transcript,
    speaker_balance, write_jsonl,
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", type=int, default=config.PHASE0_N_TRANSCRIPTS)
    ap.add_argument("--split", default="dev")
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--seed", type=int, default=config.SEED)
    args = ap.parse_args()

    config.ensure_dirs()
    questions = sample_questions(args.n, split=args.split, seed=args.seed)
    if len(questions) < args.n:
        print(f"WARNING: only {len(questions)} eligible questions found")

    half = len(questions) // 2
    # Balanced within each condition so that neither the option letter nor the
    # speaking order of the gold side correlates with condition or with truth.
    speaks_first = {
        "misleading": balanced_flags(half, seed=args.seed),
        "clean": balanced_flags(len(questions) - half, seed=args.seed + 1),
    }
    transcripts = []
    for i, q in enumerate(questions):
        condition = "misleading" if i < half else "clean"
        idx = i if condition == "misleading" else i - half
        transcripts.append(make_phase0_transcript(
            q,
            condition,
            n_rounds=args.rounds,
            article_chars=config.PHASE0_ARTICLE_CHARS,
            seed=args.seed + i,
            gold_speaks_first=speaks_first[condition][idx],
        ))

    # Exactly balanced within condition, not merely balanced in expectation.
    assign_balanced_letters(transcripts, seed=args.seed)

    out = config.PHASE0_DIR / "transcripts.jsonl"
    write_jsonl(transcripts, out)

    manifest = config.PHASE0_DIR / "questions.json"
    manifest.write_text(
        json.dumps([q.to_dict() for q in questions], indent=2), encoding="utf-8"
    )

    n_mis = sum(1 for t in transcripts if t.condition == "misleading")
    print(f"wrote {len(transcripts)} transcripts -> {out}")
    print(f"  misleading: {n_mis}   clean: {len(transcripts) - n_mis}   "
          f"rounds: {args.rounds}")
    for condition, (n_a, total) in sorted(letter_balance(transcripts).items()):
        print(f"  letter balance  [{condition:10s}]: gold is 'A' in {n_a}/{total}")
    for condition, (n_f, total) in sorted(speaker_balance(transcripts).items()):
        print(f"  speaker balance [{condition:10s}]: gold speaks first in {n_f}/{total}")
    print(f"  question manifest -> {manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
