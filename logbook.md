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

_GH200 run on Llama-3.1-8B-Instruct — the real Phase 0 gate._

<!-- fill in: date, gate report, position-bias diagnostic, decision to proceed or
     take the §6 fallback -->
