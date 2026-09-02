"""Drop transcripts that are not part of the current corpus plan.

The first 40-debate corpus was drawn with `sample_questions` (dev only, one random
question per article). The scaled corpus uses `sample_by_article`, which picks a
different set of articles and a different question within each. 39 of the original 40
are therefore not in the new plan -- and 26 of them sit on an article the new plan
assigns to the *opposite* condition.

Keeping those would put the same story in both the honest (probe training) and
collusion (probe test) sets, which is the leakage the article-level split exists to
prevent, and no downstream control would catch it.

Pruning is a read-only report by default:

    python scripts/phase2_prune.py                 # report
    python scripts/phase2_prune.py --apply         # rewrite, archiving what it drops

Run it *after* generation finishes -- it rewrites transcripts.jsonl, and a concurrent
append would corrupt the file. Follow it with `--finalise-only` so option letters are
rebalanced over the pruned corpus rather than over one containing orphans.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from judgeprobe import config  # noqa: E402
from judgeprobe.corpus import plan_corpus  # noqa: E402
from judgeprobe.quality import sample_by_article  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-n", type=int, default=264, help="articles in the plan")
    ap.add_argument("--per-article", type=int, default=2)
    ap.add_argument("--split", default="dev,train")
    ap.add_argument("--conditions", default="honest,collusion")
    ap.add_argument("--seed", type=int, default=config.SEED)
    ap.add_argument("--out", type=Path, default=config.PHASE2_DIR)
    ap.add_argument("--apply", action="store_true", help="rewrite the corpus")
    args = ap.parse_args()

    path = args.out / "transcripts.jsonl"
    rows = [json.loads(l) for l in path.open(encoding="utf-8") if l.strip()]

    age = time.time() - path.stat().st_mtime
    if args.apply and age < 60:
        print(f"REFUSING: {path.name} was written {age:.0f}s ago -- generation may "
              f"still be running. Rewriting it now would corrupt it.")
        return 2

    splits = tuple(s.strip() for s in args.split.split(","))
    conditions = tuple(c.strip() for c in args.conditions.split(","))
    specs = plan_corpus(
        sample_by_article(args.n, splits=splits, per_article=args.per_article,
                          seed=args.seed),
        conditions, seed=args.seed,
    )
    planned = {s.transcript_id for s in specs}
    plan_condition: dict[str, set[str]] = {}
    for s in specs:
        plan_condition.setdefault(s.question.article_id, set()).add(s.condition)

    keep = [r for r in rows if r["transcript_id"] in planned]
    drop = [r for r in rows if r["transcript_id"] not in planned]
    conflicting = [r for r in drop
                   if r["condition"] not in plan_condition.get(r["meta"]["article_id"], set())
                   and r["meta"]["article_id"] in plan_condition]

    print(f"corpus:   {len(rows)} transcripts")
    print(f"in plan:  {len(keep)}")
    print(f"to drop:  {len(drop)}   (of which {len(conflicting)} conflict with the "
          f"plan's condition for their article)")
    print(f"plan size: {len(planned)}")

    if not args.apply:
        print("\nreport only -- rerun with --apply to rewrite")
        return 0
    if not drop:
        print("\nnothing to drop")
        return 0

    archive = args.out / "pruned_transcripts.jsonl"
    with archive.open("a", encoding="utf-8") as f:
        for r in drop:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    shutil.copy2(path, path.with_suffix(".jsonl.bak"))
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in keep) + "\n",
                    encoding="utf-8")
    print(f"\ndropped {len(drop)} -> {archive}")
    print(f"backup of the old corpus -> {path.with_suffix('.jsonl.bak')}")
    print(f"corpus now {len(keep)} transcripts")
    print("next: python scripts/phase2_build_corpus.py --finalise-only")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
