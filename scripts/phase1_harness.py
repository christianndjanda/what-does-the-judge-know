"""Phase 1 -- run the judge harness over a transcript corpus.

One entry point for every later phase's forward passes: Phase 2 batch-judges its
debates with this, Phase 3 reads the activations it wrote, Phase 5 reruns it with
`--rounds 1,2,full`. Resumable, streams to disk, and holds no arrays in memory.

Usage:
    python scripts/phase1_harness.py --model Qwen/Qwen2.5-32B-Instruct
    python scripts/phase1_harness.py --model <id> --rounds 1,2,full --stride 4
    python scripts/phase1_harness.py --transcripts data/phase2/transcripts.jsonl \
                                     --out data/phase2/acts
    python scripts/phase1_harness.py --summary-only --out data/phase2/acts  # no GPU
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from judgeprobe import config  # noqa: E402
from judgeprobe.harness import run_corpus, summarise, write_verdicts  # noqa: E402
from judgeprobe.store import ActivationStore  # noqa: E402
from judgeprobe.transcripts import read_jsonl  # noqa: E402


def parse_rounds(spec: str) -> tuple:
    """"full" -> (None,); "1,2,full" -> (1, 2, None)."""
    out = []
    for part in spec.split(","):
        part = part.strip().lower()
        out.append(None if part in ("full", "all", "none") else int(part))
    return tuple(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=config.JUDGE_MODEL_MAIN)
    ap.add_argument("--transcripts", type=Path,
                    default=config.PHASE0_DIR / "transcripts.jsonl")
    ap.add_argument("--out", type=Path, default=config.DATA_DIR / "phase1" / "acts")
    ap.add_argument("--rounds", default="full",
                    help='truncations to capture: "full", or "1,2,full" for Phase 5')
    ap.add_argument("--stride", type=int, default=1,
                    help="cache every Nth layer (the last is always kept)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--no-resume", action="store_true",
                    help="rejudge transcripts already in the store")
    ap.add_argument("--include-article", action="store_true",
                    help="ablation: let the judge see the passage excerpt")
    ap.add_argument("--summary-only", action="store_true",
                    help="recompute verdict metrics from an existing store; no model")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    rounds = parse_rounds(args.rounds)

    if args.summary_only:
        store = ActivationStore(args.out)
        summary = summarise(store)
        print(json.dumps(summary, indent=2))
        print(f"verdicts -> {write_verdicts(store, args.out / 'verdicts.json')}")
        return 0

    transcripts = read_jsonl(args.transcripts)
    print(f"{len(transcripts)} transcripts from {args.transcripts}")
    print(f"model: {args.model}   rounds: {args.rounds}   stride: {args.stride}")

    from judgeprobe.judge import Judge  # imported late: loading torch is slow

    judge = Judge(args.model, device=args.device, seed=config.SEED,
                  layer_stride=args.stride)
    print(f"loaded: {judge.n_layers} layers, d_model {judge.d_model}, "
          f"caching {len(judge.cached_layers)}")

    store = ActivationStore(
        args.out,
        layers=list(judge.cached_layers),
        meta={"model_id": judge.model_id, "rounds": args.rounds,
              "layer_stride": judge.layer_stride,
              "transcripts": str(args.transcripts),
              "include_article": args.include_article},
    )
    report = run_corpus(judge, transcripts, store, through_rounds=rounds,
                        resume=not args.no_resume, limit=args.limit,
                        include_article=args.include_article)

    report_path = args.out / "run_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_verdicts(store, args.out / "verdicts.json")

    print(json.dumps(report["verdicts"], indent=2))
    print(f"{report['n_judged']} judged in {report['elapsed_s']}s, "
          f"{report['store_bytes'] / 1e6:.1f} MB -> {store.root}")
    print(f"report -> {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
