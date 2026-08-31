# debate-judge-probe

When an adversary successfully steers an LLM judge to the wrong verdict, does the
judge's residual stream still linearly encode the correct answer?

See `judge-probe-design.md` for the full design. This repo currently implements
**Phase 0 — sanity checks** only.

## Layout

```
src/judgeprobe/
  config.py       paths, model ids, seeds; everything overridable by env var
  quality.py      QuALITY loading, two-option restriction, best_distractor_for_question()
  transcripts.py  Transcript/Turn types, round truncation, Phase 0 templated generator
  judge.py        Judge: verdict production + residual-stream capture from one model
  probes.py       mass-mean probe (primary), logistic (secondary), layer sweep, LOO
  store.py        incremental on-disk activation cache (streaming, resumable)
  harness.py      the corpus loop: judge -> store, resumable, verdict metrics
scripts/
  check_env.py                 environment check -- run this first on the GH200
  phase0_build_transcripts.py  Phase 0.1: ten throwaway transcripts
  phase0_gates.py              Phase 0.2-0.4: Gates A, B, C
  diagnose_position_bias.py    why does the judge prefer the first option?
  phase1_selftest.py           Phase 1: harness contracts, on CPU
  phase1_harness.py            Phase 1: run the harness over a transcript corpus
  setup_lambda.sh              bootstrap a Lambda Labs GPU instance
```

Outputs land in `data/phase0/` (gitignored): `transcripts.jsonl`, `verdicts.json`,
`activations.npz`, `gate_report.json`.

## Running Phase 1

The harness is what every later phase's forward passes go through. Check it on CPU
first -- a 135M model exercises the same code paths as the 32B one:

```bash
python scripts/phase1_selftest.py       # 24 contract checks, no GPU, ~1 minute
```

Then run it over a corpus:

```bash
python scripts/phase1_harness.py --model <hf-id>                    # full debates only
python scripts/phase1_harness.py --model <hf-id> --rounds 1,2,full  # Phase 5 truncations
python scripts/phase1_harness.py --summary-only --out data/phase1/acts   # no GPU
```

It streams one `.npy` per (round, presentation order, class) into an
`ActivationStore` and flushes the manifest after every debate, so a killed run
resumes where it stopped. `--stride N` caches every Nth layer; the final layer is
always kept. Verdict metrics -- parse rate, order consistency, accuracy, mean
P(gold) -- are recomputed from the manifest by `--summary-only`, so Q0 can be
watched while a corpus is still being built.

| contract | what it checks |
| --- | --- |
| C1 | verdict and activations come from one model, under one options layout |
| C2 | layer stride keeps every Nth layer and never drops the last |
| C3 | truncation through round *k* is a real prefix of the debate |
| C4 | activations stream to disk, round-trip exactly, and a rerun resumes |

## Running Phase 0 on Lambda Labs

```bash
# on the instance, after `git clone` and copying QuALITY into data/quality/
bash scripts/setup_lambda.sh --model <hf-id>
```

The script refuses to pip-install torch (the image's is the known-good one) and
aborts if pip replaces it anyway -- the aarch64 failure mode the design doc warns
about. It also checks disk, CUDA, the QuALITY data and HF auth before pulling any
weights. Then:

```bash
tmux new -s phase0
python scripts/phase0_build_transcripts.py
python scripts/diagnose_position_bias.py --model <hf-id>
python scripts/phase0_gates.py --model <hf-id>
```

Gate C reruns from cached activations (`--gate c`) with no GPU, so the instance can
be stopped before the analysis.

## Running Phase 0 manually

On the GH200, in order:

```bash
python scripts/check_env.py                                    # torch/CUDA/QuALITY
python scripts/check_env.py --model Qwen/Qwen2.5-32B-Instruct   # + load model
python scripts/phase0_build_transcripts.py                     # 0.1
python scripts/phase0_gates.py --model Qwen/Qwen2.5-32B-Instruct
```

`phase0_gates.py` exits non-zero if any gate fails. Gates can be run singly with
`--gate a|b|c`; Gate C reads cached activations, so it reruns without the GPU.

Environment variables: `JUDGE_MODEL`, `QUALITY_DIR`, `JUDGE_PROBE_DATA`,
`JUDGE_PROBE_LOGS`, `PHASE0_ARTICLE_CHARS`.

QuALITY v1.0.1 is resolved from `QUALITY_DIR`, then `data/quality/`, then
`~/Downloads/QuALITY.v1.0.1/`.

## Gate criteria

Thresholds are fixed in `scripts/phase0_gates.py` so they are not chosen after
seeing the numbers.

| Gate | Checks | Pass |
| --- | --- | --- |
| A | judge produces parseable verdicts, in both presentation orders | 20/20 parse |
| B | activations cache with sane shapes | shape `(n_layers, d_model)`, finite, contrast pairs differ, hooks agree with `hidden_states` |
| C | mass-mean probe beats chance on easy data | best-layer leave-one-pair-out ≥ 0.75, and ≥ 3 layers above that |

Gate B also checks the hooks against HF's `output_hidden_states`. Layers 0..n-2 must
match exactly; the last entry of `hidden_states` is post-final-norm, so it is checked
against `norm(hook[-1])` instead. The hooks keep the raw residual stream at every
layer — mixing pre- and post-norm vectors would make the layer sweep incommensurable.

Gate C's leave-one-out is **leave-one-pair-out**: a contrast pair shares its entire
context, so holding out only one half leaks. With ten pairs, in-sample accuracy is
uninformative — read the LOO column.

If Gate C fails on easy data, take the §6 fallback (single misleading argument, no
multi-turn debate) rather than scaling up.

## Decisions taken during implementation

Flagged because they are not spelled out in the design doc:

- **The judge does not see the passage.** Standard debate protocol: debaters read
  the article, the judge gets only the question, the two options and the transcript.
  A judge that can read the passage cannot be steered by rhetoric, which would make
  Q0 vacuous. `include_article=True` is available as an ablation.
- **Every debate is judged twice, with the two answers trading slots**
  (counterbalancing). The verdict counts only when both orders agree on the
  *answer*; otherwise presentation order decided it and the debate is reported as
  order-inconsistent rather than scored. Probing uses the mean activation across the
  two orders. Justified by `diagnose_position_bias.py`: the judge's slot preference
  is additive and independent of the option text, which is what a swap cancels.
- **Gold's letter is balanced within condition** across the corpus, not hashed per
  question. Hashing balances only in expectation, and at Phase 0 sizes that let a
  5/5-vs-2/5 split manufacture a fake steering effect (see logbook).
- **Every verdict also records a forced-choice `P(gold)`** from the next-token
  logits over `{A, B}`. Gate A still tests generation-and-parse as written; the
  forced-choice figure means a parse failure never costs a datapoint downstream.
- **Questions are filtered to unanimous-gold, ≥2 best-distractor votes.** Judge
  errors should come from the debate, not from genuine question ambiguity.
- **No scikit-learn.** Mass-mean and logistic are implemented in numpy/torch —
  one fewer aarch64 wheel to fight on the GH200.
- **Phase 0 runs on the 32B judge, not the doc's 8B stepping stone.** Phase 0 is
  ~60 forward passes -- seconds of GPU time either way -- so the two-stage plan
  bought about 35 cents and an extra model download, at the cost of making a Gate C
  failure ambiguous between "plumbing broken" and "model too small". Needs ~65GB for
  fp16 weights: the GH200's 96GB, or an 80GB card with less headroom.
- **Phase 0 transcripts are templated, not model-generated.** The doc allows
  artificiality here; this keeps Phase 0 free and deterministic.
