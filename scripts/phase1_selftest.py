"""Phase 1 self-test -- the harness contracts, on CPU, before any GPU time.

Phase 0's gates ask whether the *judge* behaves. These ask whether the *harness*
does, which is a different question and a much cheaper one: a 135M model on CPU
exercises every code path the 32B run will take, so a plumbing bug costs seconds
here instead of an hour of GH200 time.

The contracts, one per Phase 1 deliverable:

    C1  verdict and activations come from one model, with one options layout
    C2  layer stride caches every Nth layer and never drops the last
    C3  truncation through round k is a real prefix of the debate
    C4  activations stream to disk, round-trip exactly, and a rerun resumes

Nothing here is a result about the judge -- a 135M model's verdicts are noise. It
is a check that the machinery the results will be produced by works.

Usage:
    python scripts/phase1_selftest.py                      # SmolLM2-135M on CPU
    python scripts/phase1_selftest.py --model <id> --device cuda
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from judgeprobe import config, probes  # noqa: E402
from judgeprobe.harness import PHASE5_ROUNDS, run_corpus, summarise  # noqa: E402
from judgeprobe.store import ActivationStore  # noqa: E402
from judgeprobe.transcripts import read_jsonl  # noqa: E402

# Small enough to run on the dev box, instruct-tuned so it has a chat template --
# every prompt goes through `apply_chat_template`, so a base model would exercise a
# path the real run never takes.
SMOKE_MODEL = "HuggingFaceTB/SmolLM2-135M-Instruct"
N_TRANSCRIPTS = 3
STRIDE = 4


class Checks:
    """Collects pass/fail rows so one failure does not hide the rest."""

    def __init__(self) -> None:
        self.rows: list[dict] = []

    def check(self, contract: str, name: str, ok: bool, detail: str = "") -> bool:
        ok = bool(ok)
        self.rows.append({"contract": contract, "name": name, "pass": ok,
                          "detail": str(detail)})
        print(f"  [{'PASS' if ok else 'FAIL'}] {contract} {name}"
              + (f"  -- {detail}" if detail else ""))
        return ok

    @property
    def passed(self) -> bool:
        return all(r["pass"] for r in self.rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=SMOKE_MODEL)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--transcripts", type=Path,
                    default=config.PHASE0_DIR / "transcripts.jsonl")
    ap.add_argument("-n", type=int, default=N_TRANSCRIPTS)
    ap.add_argument("--out", type=Path,
                    default=config.DATA_DIR / "phase1" / "selftest")
    ap.add_argument("--keep", action="store_true", help="keep the smoke-test store")
    args = ap.parse_args()

    transcripts = read_jsonl(args.transcripts)[: args.n]
    if not transcripts:
        print(f"no transcripts in {args.transcripts}; run phase0_build_transcripts.py")
        return 2
    shutil.rmtree(args.out, ignore_errors=True)

    from judgeprobe.judge import Judge  # late import: torch is slow to load

    print(f"loading {args.model} on {args.device} ...")
    judge = Judge(args.model, device=args.device, seed=config.SEED,
                  layer_stride=STRIDE)
    print(f"  {judge.n_layers} layers, d_model {judge.d_model}, "
          f"caching {judge.cached_layers}")

    c = Checks()
    t = transcripts[0]

    # --- C2: layer stride -----------------------------------------------------
    print("\nC2  layer stride")
    expected = list(range(0, judge.n_layers, STRIDE))
    if expected[-1] != judge.n_layers - 1:
        expected.append(judge.n_layers - 1)
    c.check("C2", "cached layers are every Nth", judge.cached_layers == expected,
            f"stride {STRIDE}: {len(judge.cached_layers)}/{judge.n_layers} layers")
    c.check("C2", "final layer always kept",
            judge.cached_layers[-1] == judge.n_layers - 1)
    swept = probes.label_layers(
        [{"layer": i, "loo_acc": 0.5} for i in range(len(judge.cached_layers))],
        judge.cached_layers)
    c.check("C2", "swept rows map back to model layers",
            [r["model_layer"] for r in swept] == judge.cached_layers,
            f"row 1 -> layer {swept[1]['model_layer']}")

    # --- C3: truncation -------------------------------------------------------
    print("\nC3  truncation through round k")
    n_rounds = t.n_rounds
    c.check("C3", "test transcript has more than one round", n_rounds > 1,
            f"{n_rounds} rounds")
    prompts = {k: judge.judge_prompt(t, through_round=k) for k in (1, n_rounds, None)}
    tokens = {k: judge.n_prompt_tokens(p) for k, p in prompts.items()}
    c.check("C3", "a shorter truncation is a shorter prompt",
            tokens[1] < tokens[None], f"r1 {tokens[1]} < full {tokens[None]} tokens")
    c.check("C3", "cutting at the last round == the full transcript",
            prompts[n_rounds] == prompts[None])
    late_text = next(x.text for x in t.turns if x.round_index == n_rounds)
    c.check("C3", "round-1 prompt omits later rounds",
            late_text not in prompts[1] and late_text in prompts[None])
    collapsed = judge.truncations(t, PHASE5_ROUNDS)
    c.check("C3", "cutoffs past the end collapse into one capture",
            collapsed == [k for k in PHASE5_ROUNDS if k is not None and k < n_rounds]
            + [None],
            f"{PHASE5_ROUNDS} on a {n_rounds}-round debate -> {collapsed}")

    # --- C1: coupling ---------------------------------------------------------
    print("\nC1  verdict and activations from one model")
    out = judge.evaluate(t, through_rounds=PHASE5_ROUNDS)
    c.check("C1", "output carries the judge's model id",
            out.model_id == judge.model_id)
    c.check("C1", "verdict and activations are for the same transcript",
            out.transcript_id == out.verdict.transcript_id == t.transcript_id)
    layout = judge.layout(t)
    c.check("C1", "verdict layout matches the contrast prompts",
            out.verdict.gold_letter == layout.gold_letter)
    verdict_prompt = judge.judge_prompt(t)
    contrast = judge.contrast_prompts(t)
    c.check("C1", "each contrast prompt extends the verdict prompt",
            all(p.startswith(verdict_prompt) for p in contrast.values()),
            "same context; they differ only in the appended answer")
    c.check("C1", "both presentation orders captured, at every round",
            all(set(out.activations[rk]) == {"forward", "reverse"}
                for rk in out.round_keys))

    # --- shapes and sanity: Gate B's checks, now on the harness path -----------
    print("\nC1  array shapes and sanity")
    shapes = {a.shape for orders in out.activations.values()
              for cls in orders.values() for a in cls.values()}
    c.check("C1", "shape is (n_cached_layers, d_model)",
            shapes == {(len(judge.cached_layers), judge.d_model)}, str(shapes))
    arrays = out.to_arrays()
    c.check("C1", "one array per round x order x class",
            len(arrays) == len(out.round_keys) * 4,
            f"{len(arrays)} arrays for rounds {out.round_keys}")
    c.check("C1", "all finite", all(np.isfinite(a).all() for a in arrays.values()))
    c.check("C1", "gold and distractor differ",
            all(not np.array_equal(orders[o]["gold"], orders[o]["distractor"])
                for orders in out.activations.values() for o in orders))
    c.check("C1", "a truncation differs from the full transcript",
            all(not np.array_equal(out.activations[rk]["forward"]["gold"],
                                   out.activations["full"]["forward"]["gold"])
                for rk in out.round_keys if rk != "full"))

    # --- C4: streaming store --------------------------------------------------
    print("\nC4  streaming, resumable store")
    store = ActivationStore(args.out, layers=list(judge.cached_layers),
                            meta={"model_id": judge.model_id, "selftest": True})
    report = run_corpus(judge, transcripts, store, through_rounds=PHASE5_ROUNDS,
                        log=lambda *a: None)
    c.check("C4", "every transcript written", len(store) == len(transcripts),
            f"{len(store)} items, {report['store_bytes'] / 1e6:.2f} MB")
    c.check("C4", "the run report is JSON-serialisable",
            isinstance(json.dumps(report), str),
            "no arrays leak out of the loop")

    reloaded = ActivationStore(args.out)
    stacked = reloaded.load_stacked("full__forward__gold")
    c.check("C4", "load_stacked returns (n, n_layers, d_model)",
            stacked.shape == (len(transcripts), len(judge.cached_layers),
                              judge.d_model), str(stacked.shape))
    c.check("C4", "the round trip is exact",
            np.array_equal(
                reloaded.load(t.transcript_id, "full__forward__gold"),
                out.activations["full"]["forward"]["gold"].astype(np.float32)))
    rerun = run_corpus(judge, transcripts, reloaded, through_rounds=PHASE5_ROUNDS,
                       log=lambda *a: None)
    c.check("C4", "a rerun resumes rather than recomputes",
            rerun["n_judged"] == 0 and rerun["n_skipped"] == len(transcripts))
    summary = summarise(reloaded)
    c.check("C4", "verdict metrics recompute from the manifest alone",
            summary["n"] == len(transcripts) and 0.0 <= summary["mean_p_gold"] <= 1.0,
            json.dumps(summary))

    report_path = args.out.parent / "selftest_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(
        {"model": args.model, "device": args.device, "stride": STRIDE,
         "n_transcripts": len(transcripts), "rounds": list(out.round_keys),
         "checks": c.rows, "pass": c.passed, "verdicts": summary}, indent=2),
        encoding="utf-8")

    if not args.keep:
        shutil.rmtree(args.out, ignore_errors=True)

    n_fail = sum(1 for r in c.rows if not r["pass"])
    print(f"\n{'PASS' if c.passed else 'FAIL'}: {len(c.rows) - n_fail}/{len(c.rows)} "
          f"checks  ->  {report_path}")
    return 0 if c.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
