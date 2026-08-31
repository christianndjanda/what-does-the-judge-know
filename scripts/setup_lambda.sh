#!/usr/bin/env bash
# Bootstrap a Lambda Labs GPU instance for Phase 0.
#
# Design doc s0 warns that aarch64 wheel availability is the most common early
# time-sink on GH200. On an x86 instance (H100/A100/A10) that risk is moot, but
# the precautions cost nothing and the script keeps them either way:
#   * it NEVER pip-installs torch -- the Lambda image ships one that works
#   * it records the torch version before and after installing anything, and
#     shouts if pip swapped it out from under us
#   * the dependency list is deliberately tiny (transformers, accelerate, numpy),
#     all pure-Python or with prebuilt wheels on both architectures
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
ARCH="$(uname -m)"
echo "arch $ARCH"
if [[ "$ARCH" == "aarch64" ]]; then
  echo "aarch64: the design doc's wheel warning applies. Do not build from source;"
  echo "find an alternative dependency instead."
else
  echo "x86: the doc's aarch64 wheel risk does not apply here."
fi
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv || {
  echo "nvidia-smi failed -- is this a GPU instance?" >&2; exit 1; }
df -h . | tail -1
echo "note: 8B in fp16 needs ~16GB of weight cache; a 32B needs ~65GB."
# 32B fp16 is ~65GB of weights; with KV cache and activations at batch 1 that is
# ~68GB in use, so it needs an 80GB card. Fail here rather than after a 65GB pull.
VRAM_MB="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)"
echo "gpu memory: ${VRAM_MB} MiB"
case "$MODEL" in
  *32B*|*32b*|*27b*|*27B*)
    if (( VRAM_MB < 70000 )); then
      echo "STOP: $MODEL in fp16 needs ~68GB in use; this card has ${VRAM_MB} MiB." >&2
      echo "Use an 80GB+ card, or pick a smaller judge (JUDGE_MODEL=...)." >&2
      echo "Do not reach for quantisation: it perturbs the activations being probed." >&2
      exit 1
    fi ;;
esac

say "python"
python -c 'import sys; print(sys.version)'
python - <<'EOF'
import sys
if sys.version_info < (3, 9):
    sys.exit(f"STOP: need Python >= 3.9, got {sys.version_info[:2]}")
print("python version ok")
EOF

say "model cache location"
# Lambda instances do not survive termination, so a 65GB download repeats every
# time unless the HF cache lives on an attached persistent filesystem. If one is
# mounted, point HF_HOME at it.
if [[ -n "${HF_HOME:-}" ]]; then
  echo "HF_HOME=$HF_HOME"
else
  FS_MOUNT="$(find /home/ubuntu -maxdepth 1 -type d -not -name ubuntu -not -name '.*' 2>/dev/null | head -1)"
  if [[ -n "$FS_MOUNT" && -w "$FS_MOUNT" ]]; then
    echo "A persistent filesystem may be mounted at $FS_MOUNT."
    echo "To avoid re-downloading ~65GB after every termination, run:"
    echo "  echo 'export HF_HOME=$FS_MOUNT/hf' >> ~/.bashrc && source ~/.bashrc"
  else
    echo "HF_HOME unset and no persistent filesystem detected: weights will be"
    echo "re-downloaded (~65GB for a 32B) if this instance is terminated."
  fi
fi

say "torch that ships with the image (do not replace it)"
TORCH_BEFORE="$(python -c 'import torch; print(torch.__version__)')"
echo "torch $TORCH_BEFORE"
python -c 'import torch; assert torch.cuda.is_available(), "CUDA not available"; print("cuda ok")'

say "project dependencies"
# torch is deliberately absent from requirements.txt -- the image's build is the
# known-good one. numpy is too, and for a less obvious reason: the shipped torch is
# compiled against a specific numpy ABI, so `pip install --upgrade numpy` pulls
# numpy 2.x and silently breaks it. torch keeps its version string and stops being
# able to convert tensors, which is why checking the version string is not enough.
TORCH_BEFORE="$(python -c 'import torch; print(torch.__version__)' 2>/dev/null || echo MISSING)"
echo "torch before: $TORCH_BEFORE"

# Exercises what actually matters: the torch<->numpy ABI, and CUDA.
torch_healthy() {
  python - >/dev/null 2>&1 <<'EOF'
import torch
torch.zeros(1).numpy()
assert torch.cuda.is_available()
EOF
}

# No --upgrade: install what is missing, do not churn what works.
python -m pip install --quiet transformers accelerate "jinja2>=3.1" "Pillow>=9.1" || true

if ! torch_healthy; then
  echo "torch is unhealthy after installing dependencies."
  echo "Most likely numpy was upgraded past the ABI the shipped torch was built for."
  echo "Pinning numpy<2 and retrying..."
  python -m pip install --quiet "numpy<2"
  if ! torch_healthy; then
    echo "STOP: torch still broken after pinning numpy<2." >&2
    python -c 'import torch; torch.zeros(1).numpy()' || true
    exit 1
  fi
  echo "recovered."
fi

TORCH_AFTER="$(python -c 'import torch; print(torch.__version__)')"
if [[ "$TORCH_BEFORE" != "$TORCH_AFTER" ]]; then
  echo "STOP: pip replaced torch ($TORCH_BEFORE -> $TORCH_AFTER)." >&2
  echo "The image's torch was the known-good one. Reinstall it before continuing." >&2
  exit 1
fi
python -c 'import torch, numpy, transformers, jinja2; print("torch", torch.__version__, "| numpy", numpy.__version__, "| transformers", transformers.__version__, "| jinja2", jinja2.__version__)'

say "QuALITY data"
if python -c 'import sys; sys.path.insert(0,"src"); from judgeprobe import config; config.quality_dir()' 2>/dev/null; then
  python -c 'import sys; sys.path.insert(0,"src"); from judgeprobe import config; print("found:", config.quality_dir())'
else
  cat <<'EOF'
NOT FOUND. QuALITY v1.0.1 is not redistributed in this repo. Either:

  * copy it up from your laptop (~27MB, a few seconds):
      scp -r <local>/QuALITY.v1.0.1 ubuntu@<instance-ip>:~/what-does-the-judge-know/data/quality
  * or fetch the release from https://github.com/nyu-mll/quality and unzip it
    into data/quality/

Then re-run this script.
EOF
  exit 1
fi

say "Hugging Face auth"
# Only gated repos need a login. Qwen2.5 is Apache-2.0 and downloads anonymously,
# so do not block on auth the run does not need.
case "$MODEL" in
  meta-llama/*|google/gemma*|mistralai/*) GATED=1 ;;
  *) GATED=0 ;;
esac
if (( GATED )); then
  echo "$MODEL is a GATED repo. Accept its licence on the model page with the same"
  echo "HF account, then authenticate here."
  if python -c 'from huggingface_hub import HfApi; HfApi().whoami()' >/dev/null 2>&1; then
    python -c 'from huggingface_hub import HfApi; print("logged in as", HfApi().whoami()["name"])'
  else
    echo "not logged in. Run:  huggingface-cli login   (or export HF_TOKEN=...)" >&2
    exit 1
  fi
else
  echo "$MODEL is not gated -- no HF login required."
fi

say "preflight (a few MB, before the ~65GB weight pull)"
# Everything here is cheap and catches a class of failure that would otherwise
# surface after a 65GB download at $4.29/hr: a broken chat template, multi-token
# A/B labels, or -- as happened on Lambda Stack 22.04 -- a stale system package
# that makes the model class fail to import at all.
python - "$MODEL" <<'EOF'
import sys
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

model_id = sys.argv[1]

# Resolve and import the model class without instantiating it. This walks the same
# import chain from_pretrained uses, so a stale Pillow/jinja2 fails here for free
# rather than after the weights are on disk.
cfg = AutoConfig.from_pretrained(model_id)
cls = AutoModelForCausalLM._model_mapping[type(cfg)]
print("model class imports:", cls.__name__)

tok = AutoTokenizer.from_pretrained(model_id)
msgs = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]
text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
assert text, "empty chat template"
for letter in ("A", "B"):
    ids = tok.encode(letter, add_special_tokens=False)
    assert len(ids) == 1, f"{letter!r} is not a single token: {ids}"
print("chat template ok; A/B are single tokens")
EOF

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
