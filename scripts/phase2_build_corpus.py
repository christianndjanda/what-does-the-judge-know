"""Phase 2.1 -- generate the debate corpus (design doc s2, first half).

API-bound and GPU-free by design: the doc splits Phase 2 so the whole corpus can be
built with the GH200 stopped, then judged in one sitting by `phase1_harness.py`.

Start with a dry run. It plans the corpus, checks the balances, renders real prompts
to a file and estimates the cost, all without a single API call:

    python scripts/phase2_build_corpus.py --dry-run -n 40

Then generate:

    python scripts/phase2_build_corpus.py -n 40 --workers 8
    python scripts/phase2_build_corpus.py -n 40 --conditions collusion,relaxed
    python scripts/phase2_build_corpus.py --finalise-only     # rebalance letters

Reruns resume: a debate already in transcripts.jsonl, leaked.jsonl or failures.jsonl
is skipped.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from judgeprobe import config  # noqa: E402
from judgeprobe.corpus import CorpusWriter, build_corpus, finalise, plan_corpus  # noqa: E402
from judgeprobe.debate import (  # noqa: E402
    DebateConfig, debater_system, estimate_debate_cost, make_client, turn_prompt,
)
from judgeprobe.quality import sample_questions  # noqa: E402
from judgeprobe.transcripts import GOLD_SIDE  # noqa: E402


def dry_run(specs, cfg: DebateConfig, out_dir: Path) -> dict:
    """Plan, balance, price and render -- without touching the API."""
    per_condition: dict[str, list] = {}
    for s in specs:
        per_condition.setdefault(s.condition, []).append(s)

    print(f"\n{len(specs)} debates planned, {cfg.n_rounds} rounds each")
    for condition, group in sorted(per_condition.items()):
        first = sum(s.gold_speaks_first for s in group)
        print(f"  {condition:10s} {len(group):4d} debates   "
              f"gold speaks first in {first}/{len(group)}")

    shared = {s.question.question_id for s in specs if s.condition == "honest"} & {
        s.question.question_id for s in specs if s.condition != "honest"}
    print(f"  questions shared across conditions: {len(shared)}"
          + ("  <-- paired; see plan_corpus docstring before training a probe"
             if shared else "  (disjoint -- Phase 3's split stays clean)"))

    est = [estimate_debate_cost(s.question, cfg=cfg) for s in specs]
    total = sum(e["cost_usd"] for e in est)
    calls = sum(e["calls"] for e in est)
    print(f"\nestimated: {calls} API calls, ${total:.2f} on {cfg.model}"
          f"  (${total / len(specs):.3f} per debate)")
    print("  estimate assumes the story is cached per debater; it is a pre-flight "
          "figure, the run reports measured usage")

    # Render one real prompt set so the wording can be read before it is paid for.
    sample = specs[0]
    system = debater_system(sample.question, side=GOLD_SIDE,
                            condition=sample.condition, words=cfg.words_per_turn)
    story, instructions = system[0]["text"], system[1]["text"]
    path = out_dir / "dry_run_prompts.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"=== {sample.transcript_id} | side=gold | condition={sample.condition} ===\n\n"
        f"--- system block 1 (cached, {len(story)} chars) ---\n{story[:1500]}\n"
        f"...[{len(story) - 1500} more chars of story]...\n\n"
        f"--- system block 2 ---\n{instructions}\n\n"
        f"--- first user turn ---\n"
        + turn_prompt("", round_index=1, n_rounds=cfg.n_rounds,
                      speaker="Debater A", words=cfg.words_per_turn),
        encoding="utf-8")
    print(f"\nsample prompts -> {path}")
    return {"n_debates": len(specs), "calls": calls, "est_cost_usd": round(total, 2)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", type=int, default=40, help="questions to sample")
    ap.add_argument("--conditions", default="honest,collusion",
                    help="comma-separated: honest, collusion, relaxed")
    ap.add_argument("--rounds", type=int, default=config.PHASE2_ROUNDS)
    ap.add_argument("--model", default=config.DEBATER_MODEL)
    ap.add_argument("--split", default="dev")
    ap.add_argument("--seed", type=int, default=config.SEED)
    ap.add_argument("--out", type=Path, default=config.PHASE2_DIR)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--words", type=int, default=150, help="word cap per turn")
    ap.add_argument("--paired", action="store_true",
                    help="every question in every condition (breaks Phase 3's split)")
    ap.add_argument("--no-classify", action="store_true",
                    help="skip the covertness side-classifier (one call per debate)")
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--limit", type=int, default=None, help="cap debates this run")
    ap.add_argument("--dry-run", action="store_true", help="plan and price, no calls")
    ap.add_argument("--finalise-only", action="store_true",
                    help="rebalance option letters over an existing corpus")
    args = ap.parse_args()

    config.ensure_dirs()
    writer = CorpusWriter(args.out)

    if args.finalise_only:
        print(json.dumps(finalise(writer, seed=args.seed), indent=2))
        return 0

    cfg = DebateConfig(model=args.model, n_rounds=args.rounds, words_per_turn=args.words,
                       classify=not args.no_classify)
    conditions = tuple(c.strip() for c in args.conditions.split(","))
    questions = sample_questions(args.n, split=args.split, seed=args.seed)
    if len(questions) < args.n:
        print(f"WARNING: only {len(questions)} eligible questions in {args.split}")
    specs = plan_corpus(questions, conditions, seed=args.seed, paired=args.paired)
    if args.limit is not None:
        specs = specs[: args.limit]

    if args.dry_run:
        report = dry_run(specs, cfg, args.out)
        (args.out / "dry_run.json").write_text(json.dumps(report, indent=2),
                                               encoding="utf-8")
        return 0

    if not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")
            or (Path.home() / ".config" / "anthropic").exists()):
        print("no Anthropic credentials found. Set ANTHROPIC_API_KEY, or run "
              "`ant auth login`. Re-run with --dry-run to plan without calling.")
        return 2

    print(f"{len(specs)} debates, {conditions}, {cfg.n_rounds} rounds, "
          f"{args.workers} workers, model {cfg.model}")
    client = make_client()
    report = build_corpus(client, specs, writer, cfg=cfg, workers=args.workers,
                          resume=not args.no_resume)
    report["balance"] = finalise(writer, seed=args.seed)

    path = args.out / "generation_report.json"
    path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(json.dumps({k: report[k] for k in
                      ("counts", "covertness", "covert_rate", "usage")}, indent=2))
    print(f"corpus -> {writer.transcripts}\nreport -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
