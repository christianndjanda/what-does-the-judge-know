"""Corpus-level judge harness (design doc Phase 1).

`Judge.evaluate` handles one debate. This is the loop around it: read transcripts,
judge each one, stream its activations to an `ActivationStore`, and keep enough
bookkeeping that a killed run resumes rather than restarts.

Three properties the Phase 2-5 runs depend on:

* **Resumable.** The store's manifest is flushed after every debate, so `has()`
  tells a rerun what is already done. A 600-debate Phase 5 run at ~16 MB and a few
  seconds each is long enough that losing it to one OOM or one dropped SSH session
  is a real cost.
* **Nothing accumulates.** Only counters and per-debate verdict dicts stay in
  memory; the arrays go straight to disk. The verdict dicts are small (~1 KB) and
  are what Q0 is computed from.
* **Verdicts live in the manifest.** `summarise` recomputes the Q0 numbers from an
  interrupted run's store without a GPU, so the steering rate can be watched while
  the corpus is still being built -- which is the stop condition the doc attaches
  to Phase 2.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from .judge import Judge
from .store import ActivationStore
from .transcripts import Transcript

# Phase 5 wants round 1, round 2 and the full debate. `Judge.truncations` collapses
# a cutoff that lands past the end, so a 2-round debate captures r1 and full.
PHASE5_ROUNDS = (1, 2, None)


def item_info(out, transcript: Transcript, seconds: float) -> dict:
    """The per-debate record written alongside the arrays."""
    return {
        "transcript_id": transcript.transcript_id,
        "question_id": transcript.question_id,
        "condition": transcript.condition,
        "n_rounds": transcript.n_rounds,
        "gold_letter": out.verdict.gold_letter,
        "gold_speaks_first": transcript.meta.get("gold_speaks_first"),
        "round_keys": list(out.round_keys),
        "model_id": out.model_id,
        "seconds": round(seconds, 2),
        "verdict": out.verdict.to_dict(),
    }


def run_corpus(
    judge: Judge,
    transcripts: list[Transcript],
    store: ActivationStore,
    *,
    through_rounds=(None,),
    resume: bool = True,
    limit: int | None = None,
    log=print,
    **kw,
) -> dict:
    """Judge every transcript, streaming activations into `store`.

    Returns a run summary. `kw` goes through to `Judge.evaluate` (e.g.
    `include_article=True` for the ablation).
    """
    todo = [t for t in transcripts
            if not (resume and store.has(t.transcript_id))]
    skipped = len(transcripts) - len(todo)
    if limit is not None:
        todo = todo[:limit]
    if skipped:
        log(f"resuming: {skipped} already in {store.root}, {len(todo)} to go")

    started = time.time()
    for i, t in enumerate(todo, 1):
        t0 = time.time()
        out = judge.evaluate(t, through_rounds=through_rounds, **kw)
        arrays = out.to_arrays()
        store.write(t.transcript_id, arrays,
                    info=item_info(out, t, time.time() - t0))
        done = time.time() - started
        eta = done / i * (len(todo) - i)
        v = out.verdict
        log(f"[{i}/{len(todo)}] {t.transcript_id}  "
            f"{'consistent' if v.consistent else 'order-split'} "
            f"choice={v.choice or '-':10s} p_gold={v.p_gold:.2f}  "
            f"{len(arrays)} arrays  {time.time() - t0:.1f}s  eta {eta / 60:.1f}m")

    return {
        "model_id": judge.model_id,
        "n_transcripts": len(transcripts),
        "n_judged": len(todo),
        "n_skipped": skipped,
        "elapsed_s": round(time.time() - started, 1),
        "layers": list(judge.cached_layers),
        "layer_stride": judge.layer_stride,
        "store": str(store.root),
        "store_bytes": store_bytes(store),
        "verdicts": summarise(store),
    }


def store_bytes(store: ActivationStore) -> int:
    return sum(p.stat().st_size for p in store.root.glob("*.npy"))


def summarise(store: ActivationStore) -> dict:
    """Q0-style verdict metrics, recomputed from the manifest. No GPU needed.

    `accuracy` is over the debates where the two presentation orders agreed on an
    answer; `forced_accuracy` uses the order-averaged P(gold), which is defined for
    every debate and so does not silently drop the order-split ones. Report both --
    the Phase 0 diagnostics showed the gap between them is the judge's position
    dependence, not noise.
    """
    rows = [r["verdict"] for r in store.info_table() if "verdict" in r]
    if not rows:
        return {"n": 0}
    consistent = [r for r in rows if r["consistent"]]
    parsed = [r for r in rows
              if r["forward"]["parsed"] and r["reverse"]["parsed"]]
    return {
        "n": len(rows),
        "parse_rate": len(parsed) / len(rows),
        "consistency_rate": len(consistent) / len(rows),
        "n_consistent": len(consistent),
        "accuracy": (sum(r["correct"] for r in consistent) / len(consistent)
                     if consistent else None),
        "forced_accuracy": sum(r["forced_choice"] == "gold" for r in rows) / len(rows),
        "mean_p_gold": sum(r["p_gold"] for r in rows) / len(rows),
        "mean_first_slot_rate": sum(r["first_slot_rate"] for r in rows) / len(rows),
    }


def write_verdicts(store: ActivationStore, path: Path) -> Path:
    """Dump the manifest's verdict records to a standalone JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"summary": summarise(store), "items": store.info_table()},
                   indent=2),
        encoding="utf-8",
    )
    return path
