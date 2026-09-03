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
  debate.py       Phase 2 debaters: prompts, conditions, leak check, covertness
  corpus.py       Phase 2 runner: plan, parallel generate, quarantine, balance
  analysis.py     Phases 3-5: what counts as steered, and the article-disjoint split
scripts/
  check_env.py                 environment check -- run this first on the GH200
  phase0_build_transcripts.py  Phase 0.1: ten throwaway transcripts
  phase0_gates.py              Phase 0.2-0.4: Gates A, B, C
  diagnose_position_bias.py    why does the judge prefer the first option?
  phase1_selftest.py           Phase 1: harness contracts, on CPU
  phase1_harness.py            Phase 1: run the harness over a transcript corpus
  phase2_selftest.py           Phase 2: pipeline contracts against a stub client
  phase2_build_corpus.py       Phase 2: generate the debate corpus (API, no GPU)
  phase2_q0.py                 Q0: the steering rate, per condition
  phase3_probe.py              Phase 3: truth probe, verdict-probe control, Q1
  phase5_trajectory.py         Phase 5: Q2 across rounds, with the length control
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

## Running Phase 2 (no GPU)

Debate generation is API-bound, so the doc splits Phase 2: build the corpus with the
instance stopped, then batch-judge it with `phase1_harness.py`.

```bash
python scripts/phase2_selftest.py                       # 28 contracts, no API calls
python scripts/phase2_build_corpus.py --dry-run -n 40   # plan, balance, price
export ANTHROPIC_API_KEY=...                            # or `ant auth login`
python scripts/phase2_build_corpus.py -n 40 --workers 8
python scripts/phase1_harness.py --model <hf-id>     --transcripts data/phase2/transcripts.jsonl --out data/phase2/acts
```

Three conditions. `honest` is Phase 3's training set (judge usually right, so truth
and verdict agree); `collusion` is both-know covert steering toward the best
distractor, and its successes are the test set; `relaxed` is the same without the
covertness requirement -- the doc's contingency lever, to be pulled only if
`collusion` yields too few judge failures, and reported separately if it is.

Conditions get **disjoint questions** by default: Phase 3 trains on honest and tests
on colluded transcripts, so a shared question would put the same answer strings on
both sides of the split. `--paired` overrides this and must not be used for probe
training.

Cost is ~$0.07 per 2-round debate on `claude-sonnet-4-6` with the story cached, so
~$3 per 40 debates. `--dry-run` prices a run before it happens and writes a sample of
the real prompts to `dry_run_prompts.txt`.

Reruns resume. Debates whose text mentions the setup ("as instructed, I will argue
weakly") are quarantined to `leaked.jsonl` rather than dropped, and failures to
`failures.jsonl`, so both rates stay recoverable.

## Running Phases 3 and 5

Both read cached activations, so neither needs a GPU -- but the store they read does
have to exist. Phase 3 runs off the full-transcript store (`acts_v2`); Phase 5 needs
the round-truncated one, which is a second judging pass.

```bash
python scripts/phase3_probe.py --acts data/phase2/acts_v2                     # primary
python scripts/phase3_probe.py --acts data/phase2/acts_v2 --criterion forced  # secondary
```

`--criterion` picks the test set, and both were fixed before the numbers existed:

| | steered means | test cases | control disagreements |
| --- | --- | --- | --- |
| `verdict` (primary) | both presentation orders agree on the distractor | 23 | 13 |
| `forced` (secondary) | order-averaged P(gold) < 0.5 | 80 | 56 |

The forced criterion is defined for every debate, so it does not condition on
order-consistency -- which Q0 showed the treatment itself degrades -- and it gives
the s3.3 verdict-probe control four times as many gold-vs-verdict disagreements to
stand on. Reports go to `phase3_report.json` and `phase3_report_forced.json`.

Phase 5 needs round-truncated activations, which is ~20 minutes on the GPU and about
5.5 GB on disk at stride 1:

```bash
python scripts/phase1_harness.py --model Qwen/Qwen2.5-32B-Instruct \
    --transcripts data/phase2/transcripts.jsonl --rounds 1,2,full \
    --out data/phase2/acts_rounds          # a FRESH directory -- see below
python scripts/phase5_trajectory.py --acts data/phase2/acts_rounds
python scripts/phase5_trajectory.py --acts data/phase2/acts_rounds --criterion forced
```

**Write it to a new directory.** Resume skips on transcript id, not on which round
keys were captured, so pointing `--rounds 1,2,full` at `acts_v2` reports "525 already
in store, 0 to go" and writes nothing. `--no-resume` is the other way out.

Every debate in the corpus is 2 rounds, so `1,2,full` collapses to two capture
points: after round 1, and the whole debate. The trajectory is a two-point comparison
rather than a curve, and the script says so.

Phase 5 trains one probe on the full transcripts of the training split and applies it
at each truncation -- the doc is explicit that the probe is not refitted per round.
It reports margins under two sign conventions: toward gold (Q2 as the doc poses it)
and toward the answer the judge actually gave (what the direction turned out to
measure). The pre-registered length-confound control -- the identical analysis on
held-out honest debates the judge got right -- is on the same axes by construction.

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
