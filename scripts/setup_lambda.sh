#!/usr/bin/env bash
# Bootstrap a Lambda Labs GPU instance for Phase 0.
#
# Design doc s0 warns that aarch64 wheel availability is the most common early
# time-sink on GH200. This script is built around that warning:
#   * it NEVER pip-installs torch -- the Lambda image ships one that works
#   * it records the torch version before and after installing anything, and
#     shouts if pip swapped it out from under us
#   * the dependency list is deliberately tiny (transformers, accelerate, numpy),
#     all pure-Python or with prebuilt aarch64 wheels
#
# Usage, on the instance:
#   bash scripts/setup_lambda.sh
#   bash scripts/setup_lambda.sh --model meta-llama/Llama-3.1-8B-Instruct
set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen2.5-32B-Instruct}"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --model) MODEL="$2"; shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

say() { printf '\n=== %s ===\n' "$1"; }

say "machine"
uname -m
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv || {
  echo "nvidia-smi failed -- is this a GPU instance?" >&2; exit 1; }
df -h . | tail -1
echo "note: 8B in fp16 needs ~16GB of weight cache; a 32B needs ~65GB."

say "torch that ships with the image (do not replace it)"
TORCH_BEFORE="$(python -c 'import torch; print(torch.__version__)')"
echo "torch $TORCH_BEFORE"
python -c 'import torch; assert torch.cuda.is_available(), "CUDA not available"; print("cuda ok")'

say "project dependencies"
# torch is deliberately absent from requirements.txt. If pip changes it anyway,
# that is the failure mode the design doc warns about -- catch it loudly.
python -m pip install --quiet --upgrade transformers accelerate numpy
TORCH_AFTER="$(python -c 'import torch; print(torch.__version__)')"
if [[ "$TORCH_BEFORE" != "$TORCH_AFTER" ]]; then
  echo "STOP: pip replaced torch ($TORCH_BEFORE -> $TORCH_AFTER)." >&2
  echo "The image's torch was the known-good one. Reinstall it before continuing." >&2
  exit 1
fi
python -c 'import transformers, accelerate, numpy; print("transformers", transformers.__version__)'

say "QuALITY data"
if python -c 'import sys; sys.path.insert(0,"src"); from judgeprobe import config; config.quality_dir()' 2>/dev/null; then
  python -c 'import sys; sys.path.insert(0,"src"); from judgeprobe import config; print("found:", config.quality_dir())'
else
  cat <<'EOF'
NOT FOUND. QuALITY v1.0.1 is not redistributed in this repo. Either:

  * copy it up from your laptop (~27MB, a few seconds):
      scp -r <local>/QuALITY.v1.0.1 ubuntu@<instance-ip>:~/what-does-the-judge-knows/data/quality
  * or fetch the release from https://github.com/nyu-mll/quality and unzip it
    into data/quality/

Then re-run this script.
EOF
  exit 1
fi

say "Hugging Face auth"
if [[ "$MODEL" == meta-llama/* ]]; then
  echo "$MODEL is a GATED repo. You must have accepted Meta's licence on the model"
  echo "page with the same HF account, then authenticate here."
fi
if python -c 'from huggingface_hub import HfApi; HfApi().whoami()' >/dev/null 2>&1; then
  python -c 'from huggingface_hub import HfApi; print("logged in as", HfApi().whoami()["name"])'
else
  echo "not logged in. Run:  huggingface-cli login   (or export HF_TOKEN=...)"
  exit 1
fi

say "environment check + model smoke test"
python scripts/check_env.py --model "$MODEL"

cat <<EOF

=== ready ===
Next, in a tmux session so an SSH drop does not kill the run:

  tmux new -s phase0
  python scripts/phase0_build_transcripts.py
  python scripts/diagnose_position_bias.py --model $MODEL
  python scripts/phase0_gates.py --model $MODEL

phase0_gates.py exits non-zero if any gate fails. Gate C reruns from cached
activations (--gate c) without the GPU, so you can stop the instance first.
EOF
