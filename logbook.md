# Logbook

Kept throughout, not reconstructed at the end (design doc §7). Records dead ends,
pivots, and the reasoning behind stop-gate decisions.

---

## Pre-registration (§8)

Committed **before** any results are seen. Each of these is publishable as-is, and
none is a reason to escalate the attack.

1. **Truth and verdict probe directions are inseparable** (high cosine, mutual
   cross-test success) → the judge has no output-independent truth representation
   to read. Q1 collapses; report it.
2. **Probe is at chance everywhere** → the judge does not linearly encode the
   answer in the assumed way.
3. **Steering never succeeds** (Q0 ≈ 0) → covert rhetorical steering does not
   reliably fool this judge in sequential QuALITY debate.

Also pre-registered, from §5: the length confound control (identical truncation
analysis on honest-side-wins transcripts) is run and **both curves are reported on
the same axes**, whatever they show.

Gate thresholds are fixed in `scripts/phase0_gates.py` rather than chosen after
seeing the numbers: Gate A all 2n generations parse (both presentation orders);
Gate C best-layer leave-one-pair-out ≥ 0.75 with ≥ 3 layers above threshold.

---

## Phase 0 — sanity checks

### Setup

Repo scaffolded: QuALITY loader, transcript representation with round truncation,
judge wrapper (verdict + residual-stream hooks on one model instance), mass-mean
and logistic probes, three gate scripts.

Design-doc gaps resolved during implementation, all recorded in the README:

- **Judge is blind to the passage.** The doc does not say either way. Standard
  debate protocol has the judge unable to read the article; more importantly, a
  judge that can read it cannot be steered by rhetoric, and Q0 would be vacuous.
  `include_article=True` kept as an ablation.
- **Option letters assigned per question**, originally a seeded hash of
  `question_id`. Without some such control, judge position bias is
  indistinguishable from a steering effect. Superseded during Phase 0 by balanced
  assignment plus counterbalancing — see "Why counterbalancing was added".
- **Forced-choice `P(gold)` recorded alongside every generated verdict.** Gate A
  still tests generate-and-parse as written, but downstream analysis never loses
  a datapoint to a parse failure.
- **Question filter:** unanimous untimed-annotator agreement with gold, and ≥ 2
  annotator votes for the best distractor. 1376 dev questions survive — ample for
  the ~60-debate corpus target. Judge errors should trace to the debate, not to
  question ambiguity.
- **No scikit-learn**, per the §0 aarch64 warning. Mass-mean is ten lines of numpy;
  logistic is LBFGS in torch.
- **Leave-one-*pair*-out, not leave-one-out.** A contrast pair shares its whole
  context, so holding out one half leaks the pair's context into the training
  means. At ten pairs the in-sample number is meaningless and only the LOO figure
  is read.
- **Layer-selection caveat recorded now**, before it can become a result: Gate C
  picks the best of ~30 layers on 10 pairs. That is a plumbing check, not
  evidence. Gate C therefore also requires a *band* of ≥ 3 working layers rather
  than one spike, and layer selection moves to a proper validation split in
  Phase 3.

### 0.1 — ten throwaway transcripts

Templated rather than model-generated (the doc permits artificiality here; this
keeps Phase 0 free and deterministic). Five "misleading", where the distractor
side argues in a grounded register and the gold side hedges; five "clean", the
reverse. Two rounds each, Debater A always on gold — the letter assignment in the
judge prompt is what removes the positional cue. (Originally a per-question hash;
now balanced within condition and counterbalanced at judging time. See below.)

### 0.2–0.4 — Gates A, B, C

_Local plumbing run (CPU, SmolLM2-135M) — these results are about the code, not
about the hypothesis; a 135M model says nothing about whether a judge encodes
truth._

- **Gate A: PASS (parseability).** 10/10 parsed. But the model answered `A` on all
  ten prompts — 100% position bias. Because letters are randomised, that surfaced
  as an apparent *effect*: 70% "verdict accuracy", and a forced-choice gold rate of
  100% on clean vs 40% on misleading. Both numbers are pure artefact of the
  A-always policy interacting with the letter assignment.

  Two things came out of this. First, the letter randomisation is load-bearing —
  without it this failure mode is invisible. Second, a degenerate judge passes a
  parseability-only Gate A, so a position-bias line was added to the Gate A output
  with a loud warning above 90%. It is reported, not gated, since the design doc
  scopes Gate A to parseability; a human decides. **If the real judge shows the same
  bias, Q0 is unmeasurable and the prompt must be fixed before Phase 1.** (This
  later drove the move to counterbalancing, which removes the bias from the verdict
  rather than only reporting it.)

- **Gate B: PASS.** Activations `(10, 30, 576)`, all finite, no contrast pair
  identical.

  *Correction:* this entry originally claimed the hooks matched
  `output_hidden_states` "to ~1e-6". That figure was never observed — the first
  gate run's output was lost to shell buffering and I wrote the number from
  expectation. When the check was actually run it reported a max deviation of
  4.86e+02. See the hook-indexing entry below for what that turned out to be.

- **Gate C: FAIL, as expected at this scale.** Best layer 21, leave-one-pair-out
  0.55; 0/30 layers above the 0.75 threshold; clean-only held-out 0.50. A 135M
  model that answers by position has no truth direction to find. This validated the
  failure path — the script exits non-zero and prints the §6 fallback instruction —
  rather than telling us anything about the hypothesis.

- Cost note: the first Gate C run took >15 min because the secondary logistic probe
  was being LBFGS-fitted with leave-one-out across every layer. It is now fitted
  only at the layer mass-mean selected, which is all a secondary comparison needs.
  Gate C now runs in seconds and reruns from cached activations without a GPU.

### Diagnosing the position bias

Ran five prompt variants over the Phase 0 transcripts to find out *why* the judge
prefers the first option (`scripts/diagnose_position_bias.py`, SmolLM2-135M):

| variant | mean P(first option) |
| --- | --- |
| normal | 0.805 |
| swapped (answers trade slots) | 0.811 |
| numeric (`1./2.`) | 0.807 |
| XY (`X./Y.`) | 0.861 |
| no_debate (transcript removed) | 0.673 |

Three conclusions, each of which kills or motivates a fix:

- **Swapping the answers moves it by 0.006.** The preference attaches to the slot,
  not to the text in it. Content is not entering the decision.
- **Relabelling preserves it** (`1/2` identical, `X/Y` stronger). So it is not a
  lexical prior on the token "A", and no relabelling fix will work.
- **A structural prior of 0.673 survives with no debate at all**, which the
  transcript then amplifies to 0.805. My reading is that a 135M model given 700
  tokens it cannot use regresses harder onto its default — but these numbers do not
  establish that mechanism.

Scope: six transcripts, one 135M model. A demonstration of the diagnostic, not a
claim about LLMs.

### Why counterbalancing was added

Recording the reasoning chain, because the decision changes what a "debate" is as a
unit of analysis in every later phase and the rationale should not have to be
reconstructed from the diff.

**1. The problem it solves is not noise — it is a fabricated positive result.**
The smoke run produced 70% verdict accuracy and a clean-vs-misleading split of
100% vs 40%: precisely the shape of the effect this project exists to look for,
from a model that never read a word. Randomised letters did not prevent this. They
converted a constant bias into a *random* one, and with n=5 per cell a random bias
correlates with condition often enough to matter. Randomisation makes the error
unbiased in expectation over many corpora; it does nothing for the one corpus you
actually run.

**2. Position bias biases the Phase 3.3 control toward a false positive.** This is
the reason that made it non-optional rather than merely tidy. Phase 3.3 trains a
verdict probe and asks whether its direction is separable from the truth direction,
with the doc's stated reading: *separable → a truth representation distinct from the
output → Q1 is live*. But if verdicts are driven by position, the verdict probe
learns a positional feature, which is close to orthogonal to any content direction.
The control passes, the cosine comes out low, and Q1 looks live — for entirely the
wrong reason. The failure mode points the same way as the hypothesis, which is the
dangerous kind.

**3. The diagnostic said a swap is the right instrument.** Swapping the answers
moved P(first option) by 0.006, so the preference is additive in the slot and
independent of the text occupying it. That is exactly the structure a swap cancels.
Had the bias been content-dependent, counterbalancing would not have been the fix.

**4. Alternatives, and why they were rejected.**

- *Relabelling* (`A/B` → `1/2` or `X/Y`): ruled out empirically. The bias survives
  relabelling unchanged (0.807, 0.861 vs 0.805).
- *Contextual calibration* (measure the no-debate prior, divide it out): only gets a
  third of the way — 0.805 → 0.667, against 0.499 for counterbalancing — because the
  prior measured without a debate understates the bias present with one. It is also
  a post-hoc correction to the output, so Q0 would become the flip rate of a
  judge-plus-correction, a system that exists nowhere but the analysis script.
  Retained as a robustness check in the writeup, not as the primary analysis.
- *Bigger judge*: expected to help a lot and already planned, but "should be fine at
  8B" is an assumption, and the gate exists to stop us running on assumptions.

**5. Why the cost is acceptable.** Two judge forward passes per debate instead of
one. That is local GPU work; Phase 2's expensive half is Sonnet generation, which is
untouched. Phase 5 already budgets six forward passes per debate for the truncation
analysis, so this is not a new order of magnitude.

**6. The asymmetry that makes it legitimate rather than a thumb on the scale.**
Presentation order is an artifact *we* introduced — something had to be printed
first. Removing it does not construct a new system; it recovers the judge's actual
content-driven behaviour. Q0 stays a claim about the judge. That is what separates
counterbalancing from calibration.

**What it does on SmolLM2:** order consistency **0%**, every verdict voided, and the
order-averaged `p_gold` collapses to 0.491–0.506. The fake 70%/100%/40% figures are
gone, replaced by the correct statement that this judge has no measurable content
signal. The design degrades into the right answer rather than a plausible wrong one.

Two pre-registered decisions it forced:

- **Order-inconsistent debates are reported, not silently dropped.** A debate whose
  verdict was decided by presentation order is a real measurement, and the rate is a
  cheap headline number. Q0 is computed over order-consistent debates only, with the
  inconsistency rate reported alongside.
- **Probing uses the mean activation across the two presentation orders**, giving one
  order-invariant representation per debate and leaving the probe fitting, layer
  sweep and leave-one-pair-out unchanged. The alternative — treating the two orders
  as separate training examples — needs group-aware cross-validation to stop a debate
  leaking across the split. Both single-order arrays are still cached so the effect
  of counterbalancing on the probe can be reported.

**Balanced letter assignment.** Gold's letter is now assigned corpus-wide, exactly
balanced within each condition, instead of hashed per question. This does not touch
position bias; it makes the specific 5/5-vs-2/5 confound structurally impossible.

### Deviation from §0.1: Phase 0 runs on the 32B, not the 8B

The doc plans two stages — `Llama-3.1-8B-Instruct` for Phase 0 plumbing, a ~27–32B
judge for Phases 3–5. Collapsed to one stage, on `Qwen2.5-32B-Instruct`.

Costed it first. Phase 0 issues ~61 forward passes over ~400-token prompts plus 20
short generations; at 32B on an H100-class card that is 10–20 seconds of compute,
against ~5 seconds at 8B. The real difference is weight download (65GB vs 16GB) and
model load, putting a realistic Phase 0 session at 30–40 min versus 20–25 min — on
the order of **35 cents** at a ~$1.50/hr instance rate. The two-stage plan was
buying almost nothing.

What it was costing is the interpretability of a Gate C failure. At 8B, a failure is
ambiguous between "the plumbing is broken" and "the model is too small", and those
call for opposite actions — fix it, versus proceed to the larger judge. Since the
doc's stop rule sends a Gate C failure straight to the §6 fallback, that ambiguity
could have killed the project on a scale artifact. Running the real judge from the
start makes the fallback decision trustworthy.

Cost note for the whole project, recorded because it reframes what to optimise: the
total GPU work across Phases 2–5 is on the order of 2,500 forward passes, well under
an hour of compute. Instance cost is therefore dominated by *how long it is left
running*, not by what it computes. Leaving it up for the full 16-hour budget is ~$24;
following the doc's own advice to keep the GPU off during Phase 2's API-bound
generation brings it to ~$4. That discipline is worth 5–10x more than any model
choice here.

Constraint this introduces: 32B fp16 is ~65GB, ~68GB in use with KV cache at batch
1, so it needs an 80GB card.

**Correction to the cost estimate above, from the actual Lambda console.** No GH200
was on offer, and the ~$1.50/hr figure was wrong. Real options: 1x A10 24GB $1.29,
1x A100 40GB $1.99, 1x H100 80GB $4.29, 2x H100 $8.38, 8x A100 $22.32. Only the
1x H100 80GB is a single-GPU card that fits the 32B. So the 8B-vs-32B decision costs
about **$9–10 over the project** (~$13 vs ~$4 for ~3h of disciplined GPU time), not
35 cents. Taking the H100 anyway: §0.1's own rationale is that probes are unreliable
below ~13B, and Gate C interpretability is worth ten dollars.

Two consequences of the instance actually available:

- **It is x86, not aarch64.** The §0 wheel-availability warning — the doc's biggest
  flagged early risk, budgeted 15 minutes and capable of costing hours — does not
  apply. `setup_lambda.sh` now detects the architecture and says so. The torch-guard
  precautions were kept regardless; they cost nothing.
- **Quantisation stays out of scope, but for a new reason.** The doc excluded it on
  aarch64 wheel grounds, which no longer hold on x86. It stays excluded because it
  perturbs the activations being probed, which is a measurement-fidelity argument
  and a stronger one. `setup_lambda.sh` refuses to proceed on an undersized card
  rather than offering quantisation as a way out.

### The §0 wheel warning bit us anyway, through numpy

First bootstrap on the H100 (Lambda Stack 22.04, x86, torch 2.7.0, Python 3.10)
broke torch. Not via a wheel — via numpy.

`setup_lambda.sh` ran `pip install --upgrade transformers accelerate numpy`, which
pulled numpy 2.2.6. The shipped torch was compiled against the numpy 1.x C ABI, so
`import torch` started emitting *"A module that was compiled using NumPy 1.x cannot
be run in NumPy 2.2.6"* and `Failed to initialize NumPy: _ARRAY_API not found`.

The instructive part is that **the guard I wrote to prevent exactly this did not
fire.** It compared `torch.__version__` before and after, and the version string was
still `2.7.0` — torch had not been replaced, it had been *broken underneath*. A
version string is not a health check.

Fixed by testing behaviour instead: `torch.zeros(1).numpy()` (exercises the
torch↔numpy ABI) plus `torch.cuda.is_available()`. If that fails after installing,
the script now pins `numpy<2` and re-checks, rather than reporting success. Also
dropped the blanket `--upgrade`, which had no reason to churn a working numpy in
the first place.

Generalisation worth keeping for the writeup: §0 framed the environment risk as
*wheel availability on aarch64*. The actual mechanism was an ABI break from an
unnecessary upgrade, on x86, where the doc said the risk did not apply. The lesson
is not "aarch64 is risky" but "do not upgrade what already works, and verify by
behaviour rather than by version number".

Two smaller fixes from the same run:

- The script hard-failed on missing HF auth even for `Qwen2.5-32B-Instruct`, which
  is Apache-2.0 and downloads anonymously. Auth is now required only for gated
  repos (`meta-llama/*`, `google/gemma*`, `mistralai/*`).
- Added a tokenizer-only preflight — chat template renders, `A`/`B` are single
  tokens — which is a few MB and runs before the ~65GB weight pull. Gate A's
  assumptions are now checked at $0 rather than after a download at $4.29/hr.

### Stale system packages, and how the preflight paid for itself

After the numpy fix, two more failures in the same family — Lambda Stack 22.04's
system packages are too old for transformers 5.16.1:

- `jinja2` 3.0.3 (needs ≥3.1). `apply_chat_template` renders through Jinja, so this
  broke every prompt the project builds.
- `Pillow` 9.0.1 (needs ≥9.1 for `PIL.Image.Resampling`). This one is pure
  collateral: transformers eagerly imports its object-detection loss chain, which
  touches `image_utils`, which touches Pillow. We do no vision at all, and yet
  `Qwen2ForCausalLM` would not import.

Both were missing from `requirements.txt` because they happened to be satisfied on
the dev box — the classic way a dependency stays invisible until a clean machine.

**The preflight is what made these cheap.** Both surfaced before any weights
downloaded: jinja2 during the tokenizer check, Pillow during model-class
resolution (which happens after the tiny config fetch but before the shards). Three
failed bootstrap attempts cost a few MB and a few minutes, instead of three 65GB
pulls at $4.29/hr.

Extended the preflight accordingly: it now resolves and imports the model class via
`AutoModelForCausalLM._model_mapping[type(cfg)]` without instantiating it, walking
the same import chain `from_pretrained` uses. That is the check that would have
caught the Pillow failure first time.

Standing lesson for the environment section of the writeup: on a fresh machine, the
binding constraint was not GPU, wheels, or the model — it was four stale
transitive dependencies, three of which this project never uses directly.

### A statistic that would have inverted the 32B result

Caught before the first real run, while re-reading `diagnose_position_bias.py`.

Its summary compared **means**: `|mean(normal) - mean(swapped)|`, reported as
"near 0 => the preference is for the slot, not the answer". That was right for
SmolLM2 by accident. Once letter assignment is balanced across the corpus, gold
sits in the first slot half the time, so a *content-driven* judge averages ~0.5 on
P(first) in both the normal and swapped conditions. The means cancel, the statistic
reads ~0, and the script would have announced "content-independent" precisely when
the judge was working perfectly — inverting the conclusion on the 32B.

The per-item difference does not cancel: swapping the slots flips which answer is
printed first, so a content-driven judge moves a long way on *every single item*
even though the corpus mean does not budge. Now reports
`mean(|normal_i - swapped_i|)`, plus mean P(gold) in both conditions, which is
unambiguous either way.

Re-ran on SmolLM2 to confirm the earlier conclusion survives the change: content
sensitivity 0.005, still positional. The new P(gold) row reads 0.654 / 0.346 —
summing to 1.0, the signature of a purely positional judge, since swapping flips
P(gold) exactly. A content-driven judge shows both numbers high instead.

Worth noting the near-miss: the balanced-assignment fix is what *created* this
cancellation. A correction in one place invalidated a summary statistic elsewhere,
and nothing would have flagged it — the script would simply have printed a
confident, wrong sentence.

### Hook indexing — a real bug the gate caught

Rerunning Gate B under the counterbalanced pipeline failed the hook-vs-`hidden_states`
check at 4.86e+02. Localised it: layers 0..n-2 match **exactly** (0.0), only the final
layer differs, and `model.norm(hook[-1]) == hidden_states[-1]` to 0.0.

Cause: HF appends the last entry of `hidden_states` *after* applying the final norm,
so `hidden_states[-1]` is post-norm while every other entry is not. The hooks are
correct — they capture the raw residual stream uniformly at every layer, which is
what the probing method needs. Probing a post-norm vector at the last layer and
pre-norm everywhere else would have made the layer sweep incommensurable, and the
norms differ by ~20x (935.6 vs 46.7), so this would not have been subtle downstream.

Fixed the *check*, not the capture: layers 0..n-2 must match exactly, and the final
layer must equal `norm(hook[-1])`. Both now report 0.00e+00. This is a stronger
check than the original.

### Phase 0 gate run — Qwen2.5-32B-Instruct, 1x H100 80GB, 2026-08-31

**All three gates PASS.** fp16, 64 layers, d_model 5120.

**Gate A: PASS on parseability (20/20), but the interesting number is elsewhere.**
`order_consistency_rate = 0.10`. Nine of ten debates gave *opposite answers*
depending only on which slot the options were printed in — the judge picked the
same slot in both presentations.

`first_slot_rate` was 0.55, i.e. apparently healthy, and the warning keyed to it
did not fire. Decomposing the 20 runs: 5 debates always-A, 4 always-B, 1
content-driven. The anchoring is **per-item**, so the global rate averages out and
hides it. Added a warning on order consistency, which is the sharper diagnostic;
chance for a coin-flipping judge is 0.5, so 0.10 is far below what randomness
would give.

`verdict_accuracy_consistent_only = 1.0` is on n=1 and means nothing.

**Do not read this as a finding about the judge yet.** Phase 0 transcripts are
templated boilerplate in which both debaters say near-identical things; the
conditions differ only in register. A judge with nothing to discriminate on may
default to position. This is genuinely ambiguous between "Qwen2.5-32B is heavily
position-anchored" and "our synthetic stimuli carry no signal", and Phase 0 cannot
separate them. **Re-measure on real Sonnet debates in Phase 2 before it bears on
Q0.** If order consistency is still ~0.1 there, Q0 is unmeasurable and that is a
reportable result in its own right (pre-registered negative #3).

**Gate B: PASS.** `(10, 64, 5120)` per class per order, no problems, hooks exact.

**Gate C: PASS, and strongly.** Best-layer LOO 1.0, 33/64 layers above the 0.75
threshold, clean-only held-out 1.0, and both single presentation orders 1.0
independently. It is a band, not a spike: layers 40–46 are seven consecutive 1.0s
with 0.95 shoulders. Logistic (secondary) also 1.0, cosine 0.79 with mass-mean.

Two caveats recorded before this gets treated as a result:

1. **Layer selection.** Argmax returned layer 27, which is an isolated spike
   (neighbours 0.85 and 0.90) that happened to be the first of eight ties at 1.0.
   Added `best_band`, which takes the centre of the longest contiguous run of
   top-scoring layers — layer 43 here. A spike among ~64 layers is selection noise;
   a run of seven is evidence.
2. **The probe may not need the debate at all.** The contrast pairs differ by the
   appended answer text, so a probe could separate them on properties of the answer
   strings — if QuALITY gold answers are systematically longer or more specific
   than their distractors — without reading the transcript. Gate C would score 1.0
   on a dataset artifact. Added `scripts/phase0_null_control.py`, which rebuilds the
   identical pairs with the debate removed. If null LOO stays near 1.0, Gate C is
   not evidence and the contrast-pair design needs rethinking before Phase 3.

**Standing tension to resolve in Phase 2.** Gate A says the judge's *verdict* is
mostly determined by layout; Gate C says a probe recovers the correct answer at
100% from the same forward passes. Read naively that is the project's hypothesis
appearing in the sanity checks, which is exactly when to be most suspicious. Both
numbers are on ten synthetic transcripts, and the null control has not run yet.

### Judge-ceiling diagnostic — the 0.10 was the stimuli, not the judge

Ran `scripts/diagnose_judge_ceiling.py` on Qwen2.5-32B to disambiguate Gate A's
0.10 order consistency.

| tier | consistency | accuracy | first-slot |
| --- | --- | --- | --- |
| 1 — world knowledge, no debate | **1.00** | 1.00 | 0.50 |
| 2 — same, one-line debate | 0.83 | 1.00 | 0.58 |
| 3 — Phase 0 debates + passage excerpt | 0.20 | 1.00 | 0.60 |

**Verdict: the prompt format and the judge are sound; the Phase 0 stimuli are not.**
Tier 1 at 100%, with a first-slot rate of exactly 0.50, rules out position anchoring
as a property of the format. Tier 2 shows the debate framing costs one item in six.
Tier 3 recovers the low number, so it is the templated boilerplate — both debaters
saying near-identical things — that leaves the judge nothing to discriminate on.

Consequence: **no prompt change.** The reasoning-before-answering fix under
consideration would have been tuning against a phantom. Re-measure order consistency
on real Sonnet debates in Phase 2, where it is now an interpretable metric because
tier 1 gives it a ceiling.

Also closed permanently: if Phase 2 debates still show low consistency, it is not the
format, the framing, or the judge.

Accuracy was 1.00 in every tier, including the two tier-3 debates that survived the
swap. Whenever this judge commits to a content-driven answer it is right. Small n,
but relevant to Q0 — it hints the judge may not be easy to steer.

**Prediction registered before running the null control.** On these stimuli the judge
is demonstrably not engaging with the transcript (tier 3: 0.20), yet Gate C's probe
separates gold from distractor at 1.00 LOO on the same forward passes. If the debate
is not entering the judge's decision, the probe is most likely reading the question,
the options and the appended answer string. **I therefore predict the null control
comes back high — probe still ~1.00 with the debate removed — which would mean Gate C
measures an answer-string or prior-knowledge artifact, not a judge conclusion.**
Recorded in advance so it is falsifiable: a collapse toward chance would be genuinely
surprising and would make Gate C stronger than currently credited.

### Null control — prediction wrong, Gate C survives

| | best layer | LOO |
| --- | --- | --- |
| with debate | 27 | **1.00** |
| without debate | 39 | **0.60** |

Zero of 64 layers clear 0.75 without the debate. Cosine between the two directions
at layer 27 is 0.348, and the Gate C probe scores 0.60 on null activations. The 0.60
is a maximum over 64 layers, which overstates it: with ten pairs LOO moves in steps
of 0.05, and a best-of-64 selection beats 0.5 comfortably on noise alone. The null
arm is at chance.

**The prediction registered in the previous entry was wrong.** I expected null LOO to
stay near 1.00 on the reasoning that a judge showing only 0.20 order consistency is
not engaging with the transcript, so the probe must be reading the appended answer
strings. It is not. Removing the debate destroys the separation, so Gate C is
measuring something computed from the debate. Recording this because a wrong
pre-registered prediction is the cheapest kind: it cost nothing and it removed a
live alternative explanation that would otherwise have needed arguing away in the
writeup.

The apparent contradiction it leaves — verdict mostly determined by slot order,
residual stream nonetheless carrying the answer — is not resolved, and should not be
presented as a knows-versus-says result on ten synthetic transcripts. It is a reason
to run Phase 2 and look again.

### A confound the null control cannot catch

Found while reading the generator after the null control passed: **every Phase 0
transcript had the gold side speaking first.** So "does the appended answer match
what Debater A claimed" was perfectly correlated with truth across the whole corpus.

A probe reading that structural cue would score 1.00 *and* would require the debate
to be present — so it passes the null control cleanly while measuring nothing about
truth. The null control rules out answer-string and prior-knowledge artifacts; it
says nothing about structural cues inside the transcript.

Fixed at generation time rather than post hoc, since speaking order changes the
transcript itself: `gold_speaks_first` is now balanced within each condition via
`balanced_flags`, and `speaker_balance()` reports it alongside the letter balance.
Both are 3/5 per condition, and the two are decorrelated.

**This carries into Phase 2 as a design requirement**: which debater argues the gold
answer must be randomised there too, or the same confound reappears in the real
corpus with no control able to detect it.

Gate C should be re-run on the regenerated corpus. If LOO stays at 1.00 with the
speaking order decorrelated, the result is considerably stronger than it was.

### Gate C FAILS once speaking order is decorrelated

Re-ran Gate B and C on the regenerated corpus, with `gold_speaks_first` balanced.

| | gold always first | speaker balanced |
| --- | --- | --- |
| best LOO | **1.00** | **0.60** |
| clean-only LOO | 1.00 | 0.70 |
| layers >= 0.75 | 33 / 64 | **0 / 64** |
| logistic LOO | 1.00 | 0.35 |

The whole sweep now sits between 0.35 and 0.60. **The entire Gate C signal was the
debater-position confound.** The probe was reading "does the appended answer match
what Debater A said", which was perfectly correlated with truth in the old corpus.

Worth noting `train_acc` still reaches 0.90 at layers 40-46 in the corrected run.
In-sample accuracy would have shown a strong result; leave-one-pair-out rejects it.
That is the clearest justification in this project for insisting on LOO.

**The design doc's rule would send us to the s6 fallback here. Two reasons not to
follow it yet.**

1. *The failure is confounded with stimulus quality.* The doc says "if Gate C fails
   on **easy data**". The Phase 0 corpus is not easy data -- both debaters assert the
   same kind of thing in different registers with no verifiable evidence either way,
   and the judge demonstrably does not engage with it (tier 3 order consistency 0.20
   against 1.00 on real questions). A judge that is not processing the debate has no
   considered answer in its residual stream to probe. This is the same ambiguity that
   the ceiling diagnostic already resolved once for order consistency.
2. *s6 would not rescue a method failure anyway.* The fallback probes a judge reading
   a single misleading argument versus a clean one -- still the contrast-pair method.
   If contrast pairs do not work on this judge, s6 inherits the problem. The doc's
   decision rule implicitly assumes the failure is about the debate framing rather
   than the method, and does not distinguish the two.

So the fork is: does the contrast-pair probe work on data the judge *does* engage
with? Added `scripts/phase0_gate_c_ceiling.py`: 18 world-knowledge questions the
judge answers correctly and consistently, wrapped in a one-round debate where one
side gives a real reason and the other asserts. Both nuisance factors -- option
letter and speaking order -- balanced.

- Probe separates them => the method is sound, Gate C's failure is about the Phase 0
  stimuli, do not take s6, proceed to Phase 2 and re-run Gate C on real debates.
- Probe fails there too => pre-registered negative #2, this judge does not linearly
  encode the answer in the assumed way, and s6 will not help. Report it as the
  result.

Either way it is a reportable finding, and either way it costs about three minutes.

### Gate C ceiling — the method works; the Phase 0 stimuli were the problem

18 world-knowledge items, one-round debate, option letter and speaking order both
balanced.

- best LOO **1.00**, 40/64 layers >= 0.75, band centre layer 51
- LOO 1.00 across a **26-layer contiguous run** (38-63), plus 27-28

So the contrast-pair probe separates true from false when there is something to
separate. **Gate C's failure on the Phase 0 corpus is about those stimuli, not the
method. Do not take the s6 fallback.** Proceed to Phase 2 and re-run Gate C on real
debates.

**What this does and does not establish.** It shows the measurement apparatus works:
hooks, contrast pairs, mass-mean, leave-one-pair-out can recover a truth
representation from this judge's residual stream when one is clearly present. It
does **not** show the judge forms a *debate-derived* truth representation. These are
items Qwen2.5-32B knows from pre-training -- the probe may be reading prior knowledge
rather than any conclusion drawn from the transcript. A null control on these items
would very likely still separate, precisely because the model knows the capital of
France with or without a debate.

That distinction is the actual research question, and only Phase 2's real debates
over passages the judge has not read can answer it. Recorded so the ceiling result is
not over-claimed in the writeup: it licenses proceeding, nothing more.

**An oddity worth noting.** Order consistency on these easy items was **0.50** --
chance -- while accuracy on the debates that did survive the swap was 1.00. Compare
tier 1 of the earlier ceiling diagnostic (same questions, no debate): consistency
1.00. So wrapping a trivially easy question in a debate frame halves the judge's
order robustness, even though the probe reads the answer at 100% and the judge is
right whenever it commits.

Two readings, not separated:

1. A prompt artifact of my test. The system prompt says "Two debaters have read a
   passage that you have not seen", which is false for world-knowledge items and may
   be confusing the judge.
2. A genuine knows-versus-says gap on easy items: the judge has the answer available
   internally and still lets layout decide the output.

(2) is close to the project's hypothesis, which is exactly why it should not be
claimed from this test -- it was not designed to support it, n is 18, and (1) is
unexcluded. Flagged as worth a clean experiment if time allows, not as a result.
