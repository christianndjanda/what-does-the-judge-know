"""Environment check -- design doc s0, "resolve in the first 15 minutes".

Run this FIRST on the GH200, before any project code. It answers the questions
that turn into multi-hour time sinks on aarch64:

  * is the shipped torch CUDA-enabled?
  * does the target model load in fp16 and do one forward pass?
  * are the letter tokens single tokens for this tokenizer?

Usage:
    python scripts/check_env.py                       # environment only
    python scripts/check_env.py --model <hf-id>       # + load and run the model
"""

from __future__ import annotations

import argparse
import platform
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from judgeprobe import config  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None,
                    help="HF model id to load and smoke-test (skipped if omitted)")
    args = ap.parse_args()

    print(f"python      {platform.python_version()}  ({platform.machine()}, {sys.platform})")

    import torch
    print(f"torch       {torch.__version__}")
    cuda = torch.cuda.is_available()
    print(f"cuda        {cuda}")
    if cuda:
        for i in range(torch.cuda.device_count()):
            p = torch.cuda.get_device_properties(i)
            print(f"  gpu[{i}]   {p.name}  {p.total_memory / 2**30:.1f} GiB")
    else:
        print("  WARNING: no CUDA. Phase 0 will run on CPU -- fine for plumbing with a "
              "small model, not for an 8B judge.")

    import transformers
    print(f"transformers {transformers.__version__}")

    try:
        print(f"quality     {config.quality_dir()}")
    except FileNotFoundError as e:
        print(f"quality     MISSING -- {e}")
        return 1

    if not args.model:
        print("\nenvironment OK (model smoke-test skipped; pass --model to run it)")
        return 0

    from judgeprobe.judge import Judge

    print(f"\nloading {args.model} ...")
    judge = Judge(args.model)
    print(f"  dtype={judge.dtype} device={judge.device} "
          f"layers={judge.n_layers} d_model={judge.d_model}")

    for letter in ("A", "B"):
        try:
            print(f"  token {letter!r} -> id {judge._letter_token_id(letter)}")
        except RuntimeError as e:
            print(f"  FAIL: {e}")
            return 1

    acts = judge.residuals("The capital of France is Paris.")
    print(f"  forward pass OK: residuals {acts.shape}, "
          f"norm(last layer)={float((acts[-1] ** 2).sum() ** 0.5):.1f}")
    print("\nenvironment OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
