"""Paths and defaults. Everything overridable by env var so the same code runs
on the Windows dev box and on the GH200 without edits."""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.environ.get("JUDGE_PROBE_DATA", REPO_ROOT / "data"))
LOGS_DIR = Path(os.environ.get("JUDGE_PROBE_LOGS", REPO_ROOT / "logs"))

PHASE0_DIR = DATA_DIR / "phase0"
PHASE1_DIR = DATA_DIR / "phase1"
PHASE2_DIR = DATA_DIR / "phase2"

# The design doc (s0.1) planned two stages: 8B for Phase 0 plumbing, then a
# ~27-32B judge for Phases 3-5. We run Phase 0 directly on the 32B instead.
#
# Rationale: Phase 0's inference is ~60 forward passes, which is seconds of GPU
# time either way, so the two-stage plan bought ~35 cents and one extra model
# download. What it cost was interpretability of a Gate C failure -- at 8B that is
# ambiguous between "plumbing broken" and "model too small", and those call for
# opposite actions (fix vs. proceed). Running the real judge from the start makes
# the s6 fallback decision trustworthy.
#
# Requires ~65GB for fp16 weights: the GH200's 96GB, or an 80GB card with less
# headroom. A 40GB card cannot hold this in fp16, and the doc rules out
# quantisation on aarch64.
JUDGE_MODEL_MAIN = os.environ.get("JUDGE_MODEL", "Qwen/Qwen2.5-32B-Instruct")
JUDGE_MODEL_PHASE0 = JUDGE_MODEL_MAIN
# Kept for reference: the doc's original Phase 0 model, and the 27B alternative.
JUDGE_MODEL_SMALL = "meta-llama/Llama-3.1-8B-Instruct"
JUDGE_MODEL_ALT = "google/gemma-2-27b-it"

# Phase 0 transcripts embed a truncated article excerpt -- these are plumbing
# tests, not data, and full QuALITY articles are ~5-8k tokens each.
PHASE0_ARTICLE_CHARS = int(os.environ.get("PHASE0_ARTICLE_CHARS", "4000"))
PHASE0_N_TRANSCRIPTS = 10  # 5 misleading, 5 clean (design doc s0.1)

# Debaters stay on the API (design doc s0.2); only the judge is open weights.
# The doc names claude-sonnet-4-6. Overridable -- claude-sonnet-5 is cheaper per
# token ($2/$10 against $3/$15) but is a different debater, so switching mid-corpus
# would make the transcripts non-comparable. Pick one before Phase 2 starts.
DEBATER_MODEL = os.environ.get("DEBATER_MODEL", "claude-sonnet-4-6")
PHASE2_ROUNDS = int(os.environ.get("PHASE2_ROUNDS", "2"))  # doc s2: 2-3 rounds

SEED = 20260904


def quality_dir() -> Path:
    """Resolve the QuALITY v1.0.1 directory: env var, then in-repo, then Downloads."""
    candidates = []
    if os.environ.get("QUALITY_DIR"):
        candidates.append(Path(os.environ["QUALITY_DIR"]))
    candidates += [
        DATA_DIR / "quality",
        Path.home() / "Downloads" / "QuALITY.v1.0.1",
    ]
    for c in candidates:
        if c.is_dir() and any(c.glob("QuALITY.v1.0.1*")):
            return c
    raise FileNotFoundError(
        "QuALITY v1.0.1 not found. Set QUALITY_DIR, or place the release in "
        f"{DATA_DIR / 'quality'}. Tried: {[str(c) for c in candidates]}"
    )


def ensure_dirs() -> None:
    for d in (DATA_DIR, LOGS_DIR, PHASE0_DIR, PHASE1_DIR, PHASE2_DIR):
        d.mkdir(parents=True, exist_ok=True)
