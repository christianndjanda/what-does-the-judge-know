# debate-judge-probe — Design Document

**Owner:** Christian. Christian owns experimental design, analysis, and writeup.
**Audience for this doc:** Claude Code, executing the build.
**Context:** Application project for Neel Nanda's MATS 12.0 stream. Deadline Fri Sept 4, 11:59pm PT. Task budget ~16 hours (hard max 20), plus up to 2 additional hours for the executive summary. Required deliverables are a write-up and an executive summary. A logbook is *not* required — it is kept here as a writeup aid, since this project has several stop-gates and pre-registered negative results whose rationale is worth recording as it happens.
**Relationship to prior work:** This is a white-box fork of the `debate-stego` v2 design. It reuses v2's debate setup (QuALITY, both-know collusion, multi-turn sequential, best-distractor target) as a *transcript generator only*. The research question is new.

---

## 0. Assumptions to confirm before Claude Code starts

Override any of these before execution begins.

1. **Judge model — two stages.** Phase 0 plumbing on `Llama-3.1-8B-Instruct` for fast iteration. Phases 3–5 on a ~27–32B class model (`Qwen2.5-32B-Instruct` or `gemma-2-27b-it`) in fp16, which fits in 96GB with headroom. Rationale: Marks & Tegmark found cross-dataset probe generalization unreliable below ~13B. This project trains on unsteered transcripts and tests on steered ones — a distribution shift of the same kind. Running only at 8B risks a null result that is about scale rather than about the hypothesis.
2. **Debater models:** stay on API (`claude-sonnet-4-6`). Only the judge moves to open weights.
3. **Compute:** Lambda Labs `gpu_1x_gh200` (96GB HBM3, aarch64). Single GPU — no sharding or multi-GPU complexity needed at these model sizes.
4. **Dataset:** QuALITY, two-option restriction — gold answer vs `best_distractor_for_question()`. Not all four options.
5. **Probe:** mass-mean (difference of class means), per Marks & Tegmark 2023. Logistic regression as a secondary comparison only.
6. **Corpus target:** ~60 successfully-steered debates minimum. Scale down before abandoning.
7. **Out of scope:** causal intervention/steering the judge, debater-side probing, cross-role transfer, SAEs, any fine-tuning.

**Environment warning — resolve in the first 15 minutes.**

GH200 is aarch64, not x86. Wheel availability is the most common early time-sink on this platform. Before writing any project code:

- Use the PyTorch that ships with the Lambda image rather than pip-installing a fresh one.
- Verify `torch.cuda.is_available()` and run one trivial forward pass on the target model.
- Prefer fp16 over quantised loading — 96GB is enough, and quantisation libraries are the most likely to lack aarch64 wheels.
- If a dependency has no aarch64 wheel, find an alternative rather than building from source. Compilation will eat hours you do not have.

**Note:** no v2 transcripts exist. Phase 2 is a full build, not a re-judge.

---

## 1. Research question

When an adversary successfully steers an LLM judge to the wrong verdict, does the judge's residual stream still linearly encode the correct answer?

This is a conditional, and the antecedent is an empirical bet, not a given. It is therefore measured explicitly as Q0 rather than assumed.

- **Q0 (precondition, and a result in its own right):** How often does covert collusion actually flip the judge's verdict? Measured in Phase 2. Report the rate regardless of what it is.
- **Q1 (endpoint):** Conditional on Q0 yielding cases, does a truth probe on the judge's activations outperform the judge's own verdict on those steered debates?
- **Q2 (trajectory):** Across debate rounds, does the probe signal for the correct answer stay flat (the steering *overrides* output) or decay (the steering *corrupts* the representation)?

**Why Q0 is stated separately.** If steering never succeeds, Q1 and Q2 have no test set. Burying that assumption inside the phrasing of Q1 would turn a foreseeable outcome into a silent project failure. Stated as Q0, a low steering rate is a reported finding and a rescope trigger (see Phase 2 contingency and §8).

**Note on judge scale.** The v2 judge was `claude-sonnet-4-6`. This project uses a ~32B open-weights judge, which should be easier to steer. Q0's rate is therefore not comparable to any v2 figure and should not be presented as one.

**Framing note for the writeup.** The deception here is instructed, not emergent — both debaters are told the steering target. What is not experimenter-authored is the deceptive *text*: it is model-generated, constrained to evade a side-classifier, and selected for having actually worked. Prior truth-probing work stipulates both the falsehood and its surface form. State this honestly; do not overclaim emergence.

---

## PHASE 0 — Sanity checks (DO THIS FIRST)

**Budget: 2 hours. Do not proceed past this phase until all three gates pass.**

The purpose of Phase 0 is to fail cheaply. No corpus generation, no fork modernisation, no scale. Ten hand-made examples are enough.

### 0.1 Build ten throwaway transcripts

Hand-write or quickly generate 10 short fake debate transcripts over QuALITY questions. Five containing a misleading argument for the best distractor, five clean. Artificiality is fine — these are plumbing tests, not data.

### 0.2 Gate A — the judge runs and produces parseable verdicts

Load the judge model. Feed each transcript with a verdict prompt. Confirm:
- Output is parseable into a single option choice, reliably, across all 10.
- If parsing is flaky, fix the prompt format now. Everything downstream depends on it.

### 0.3 Gate B — activations cache with sane shapes

Register forward hooks on the residual stream. For each transcript, build the two contrast pairs:

```
[transcript] The answer is {gold}.
[transcript] The answer is {best_distractor}.
```

Cache the residual stream at the **final token** of each. Confirm:
- Shape is `(n_layers, d_model)` per example.
- No NaNs, magnitudes are plausible.
- The two pairs from the same transcript are not identical.

### 0.4 Gate C — a probe separates true from false on easy examples

Fit a mass-mean probe on the 20 contrast-pair activations (10 gold, 10 distractor). Sweep layers. Confirm:
- Some layer separates the classes above chance on the clean examples.

**If Gate C fails on easy data, nothing downstream works.** Record it in the logbook and pivot to the fallback in §6 rather than scaling up.

---

## PHASE 1 — Judge harness (3 hours)

Productionise what Phase 0 prototyped.

- Judge module that takes a transcript, returns `(verdict, activations)` from the same forward pass. This coupling is non-negotiable — the knows-vs-says gap is meaningless if the verdict and activations come from different models.
- Layer sweep utility: cache every 4th layer, select the best on a validation split later.
- Truncation support: given a transcript, return the sub-transcript through round *k*. Needed for Phase 5; build it now while the parsing code is fresh.
- Serialise activations to disk (`.npy` or `safetensors`). Do not hold a corpus in memory.

---

## PHASE 2 — Transcript corpus (4 hours)

**Split this phase in two, and keep the GPU off for the first half.** Debate generation is entirely API-bound — the debaters are Sonnet, no GPU involved. Generate and serialise all transcripts to disk with the instance stopped, then spin up the GH200 and batch-judge them. Leaving the GPU idle while waiting on API latency wastes both money and wall-clock.

- Both-know collusion, best distractor as steering target, multi-turn sequential, 2–3 rounds.
- **Parallelise generation.** Serial API calls over long QuALITY contexts will consume the entire budget.
- Log every debate with: question id, gold answer, best distractor, judge verdict, per-round transcript boundaries.
- Track the **steering success rate (Q0)** per debate. This is both a headline result and the number most likely to force a rescope.

**Stop condition and contingency.** Track the steering success rate (Q0) continuously. If after 2 hours the rate implies you cannot reach 60 steered cases within budget, do **not** immediately fall back to §6. Pull this lever first:

**Relax the covertness constraint.** The side-classifier requirement is inherited from v2, where "does this still read as honest debate" was the research question. It is not load-bearing here. Q1 and Q2 need judge *failures*; covertness only reduces the supply of them. Dropping or loosening it still leaves a stimulus that is model-authored, adversarially optimised against the judge, and selected for having worked — which is what the §1 framing note actually claims.

If you pull this lever:
- Record both success rates (covert and relaxed) in the logbook. The difference is itself a reportable number.
- State plainly in the writeup which condition the probing results come from.
- Only fall back to §6 if the relaxed condition also fails to produce enough cases.

---

## PHASE 3 — Probe training and the confound control (3 hours)

### 3.1 Split

- **Train:** transcripts where steering failed or was not attempted — i.e. the judge got it right, so truth and verdict agree.
- **Test:** successfully-steered transcripts, where truth and verdict diverge.

The adversarial distribution stays fully held out. The probe must never learn "this transcript looks manipulated."

### 3.2 Train the truth probe

Mass-mean on the contrast pairs, labels from the **gold answer**. Select layer on a validation slice of the training set.

### 3.3 Train the verdict probe — the load-bearing control

Train a second probe with labels from the **judge's actual verdict** rather than the gold answer. Then compare:

- Cosine similarity between the two directions.
- Cross-test: truth probe on the verdict task, verdict probe on the truth task.

**Interpretation:**
- Near-identical directions → the "truth probe" is only recovering the verdict direction. Q1 collapses. Report this honestly; it is a real negative result.
- Separable directions → there is a truth representation distinct from the output. Q1 is live.

**Run this before Phase 4.** It is cheap and it gates whether the headline result means anything.

---

## PHASE 4 — Q1 analysis (2 hours)

On the held-out steered set:

- Probe accuracy vs judge verdict accuracy. The gap is the headline number.
- Report the probe's accuracy specifically on debates the judge got *wrong*.
- Confidence intervals. The sample is small; do not report a bare point estimate.

---

## PHASE 5 — Q2 trajectory analysis (3 hours)

For each debate, build contrast pairs against truncated transcripts: through round 1, through round 2, full. Six forward passes per debate instead of two. Apply the **already-trained** probe — do not retrain per round.

Plot probe score for the gold answer against round index.

**Reading the curves:**
- Flat and high, wrong verdict → **override**. Judge knew throughout; steering hit the output only.
- High then decaying → **corruption**. The debate eroded the internal representation.
- Low throughout → the judge never encoded the truth.

**Length confound control (required).** Longer context may shift probe scores independent of content. Run the identical truncation analysis on transcripts where the honest side won. If those also decay, the effect is length, not steering. Report both curves on the same axes.

---

## 6. Fallback if Phase 0 or Phase 2 fails

Drop the multi-turn debate framing. Probe a judge reading a **single misleading argument** for the best distractor, versus a clean argument. This still answers Q1, still uses the contrast-pair method, and still has an adversarially-generated stimulus. It loses Q2 entirely.

This fallback is a legitimate result, not a failure. Decide to take it early rather than late.

---

## 7. Deliverables

- Executive summary (Neel's stream requires one). Format, per the stream's admissions doc: **~1 page including graphs is ideal, hard caps of 3 pages and 600 words.** Sits at the top of the writeup, not as a separate file. Recommended shape: problem statement and why it matters; high-level takeaways and the interesting findings; then one paragraph plus one graph per key experiment. Structure it around one or two insights rather than a chronological dump of every experiment. "Specifics beat vibes: name the models, the key experiment, the surprising number." And: "If bad data would sink your project, show me the data" — include *randomly selected*, not cherry-picked, qualitative examples wherever a result leans on transcript quality or on an LLM judgment.
- Main writeup: research question, method, Q1 and Q2 results, confound controls, limitations.
- Logbook kept **throughout**, not reconstructed at the end. Record dead ends and pivots. Not a submission requirement — keep it because it makes the writeup much easier, and this project has enough branch points that reconstructing the reasoning afterwards would be lossy.
- Code, cleaned enough to read.

**Reserve the final 3 hours for the writeup.** This is the most commonly regretted omission in published MATS application retrospectives.

---

## 8. Pre-registered negative results

Commit to these now, in the logbook, before seeing results. Each is publishable as-is.

- Truth and verdict probe directions are inseparable → the judge has no output-independent truth representation to read.
- Probe is at chance everywhere → the judge does not linearly encode the answer in the assumed way.
- Steering never succeeds → covert rhetorical steering does not reliably fool this judge in sequential QuALITY debate.

Do not escalate the attack to manufacture an effect.

---

## 9. Limitations to state explicitly in the writeup

- Deception is instructed, not emergent from training pressure.
- Single judge model, single dataset.
- Round-boundary token positions are an assumption about where summarisation occurs, not a validated localisation. Marks & Tegmark used activation patching to find their positions; this project does not.
- Correlational only. No causal intervention on the probe direction.
- **The setup-leak rate is higher than the quarantine reports, and the corpus was not re-filtered.** `find_leak`'s `\bi (?:was|am) (?:told|assigned)\b` requires a literal space, so it passes the contracted and possessive forms the debaters actually produced — "I'm assigned to argue X", "my assigned answer". Auditing the corpus for those forms finds **20 of 525 debates (3.8%; 17 honest, 3 collusion)** in which a debater refers to its own assignment, against the 3 (0.6%) the gate quarantined. Bounded rather than fixed: **none of the 20 is in the 23-debate steered test set**, so Q1 is untouched; **one is in the 24-debate Phase 5 control arm**; and dropping all 20 moves Q0's verdict-error gap from +0.193 to +0.189 and its forced-choice gap from +0.088 to +0.081. The concentration in the honest arm is the expected direction — a sincere distractor debater hedges and concedes — and would if anything have flattered the honest baseline. The gate was left unchanged so the corpus keeps the composition the committed results were computed on; `scripts/sample_qualitative.py` audits and annotates instead. Found by the first draw of the random qualitative sample, which is the argument for drawing it.
