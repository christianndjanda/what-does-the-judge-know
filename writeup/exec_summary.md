# The Judge's "Truth" Direction Is a Verdict Direction

**Code, per-phase logbook, and every report JSON:**
[github.com/christianndjanda/what-does-the-judge-know](https://github.com/christianndjanda/what-does-the-judge-know)
— every Q0/Q1/Q2 number below is reproducible from `data/phase2/acts_rounds/*.json`,
and the Phase 0 gates and controls from `data/phase0/*.json`. The pre-registration is
§8 of `judge-probe-design.md`, committed before any results existed.

---

## Executive summary

Debate protocols for scalable oversight assume a judge whose reasoning can be
checked. If an LLM judge still encodes the correct answer after an adversary talks
it into the wrong one, a linear probe is a cheap audit tool: read the residual
stream, catch the manipulation. I tested this on sequential debate. It fails, and
the failure is easy to miss.

**Takeaway.** A probe trained where the judge is right recovers the correct answer 91%
of the time, and tells you nothing about whether the judge has been fooled. On
debates where covert steering succeeded, the same probe is wrong 22 times out of
23. It reads the verdict, not the truth.

**Setup.** Judge: Qwen2.5-32B-Instruct, which sees the question, two options and the
transcript, never the passage. Debaters: claude-sonnet-4-6. 525 QuALITY debates,
264 colluded, 261 honest, on disjoint articles. Each debate is judged twice, options
swapped.

![Q0: covert collusion raises the judge's error rate, and more reliably destroys its verdict entirely](../figures/fig1_q0.png)

**Q0: the attack works, mostly by voiding the debate.** Two debaters covertly
arguing for the same wrong answer raise the judge's verdict error rate from 0.118
to 0.311 (p = 0.0013) and its forced-choice error rate from 0.215 to 0.303
(p = 0.013). Unexpectedly, the larger effect is that steering destroys the verdict
more reliably than it flips it: order-consistency, meaning both presentation orders
agree, falls from 0.421 to 0.280 (p = 0.00048). A colluded debate that leaves the
judge nothing decisive pushes it back onto position bias.

![Q1: 91% where truth and verdict agree, below chance where they come apart](../figures/fig2_q1.png)

**Q1: the probe reads the verdict.** A mass-mean probe trained on debates the
judge got right, layer chosen on held-out articles, reaches 0.912 validation
accuracy at layer 44; 24 of 64 layers clear 0.75, so this is a band, not a spike.
On the 23 debates where steering succeeded it gets one right (binomial p below
chance = 2.9e-6) and agrees with the judge's wrong verdict 22 times. Its cosine
with a probe trained on the verdict instead of the truth is 0.982. The two numbers
are not in tension: on the training distribution truth and verdict coincide by
construction, so a pure verdict direction scores 0.912 there. The steered set is the
only place the two labels come apart.

![The result I deleted: the near-tie effect was regression to the mean](../figures/fig3_killed.png)

**The result I deleted.** My pre-registered secondary criterion gave 80 test
cases and said the opposite: the probe read gold 74.4% of the time on debates where
the judge never committed (p = 0.001). That is the knows-versus-says gap this
project was looking for. It did not survive a clean test. The bin came from a
different probe, and conditioned on P(gold) falling just below 0.5, which selects
cases where noise ran low. Re-run with one probe across every order-split debate:
0.653, below the judge's own forced-choice accuracy of 0.709 on the same debates.
Regression to the mean, not a truth signal.

![Q2: flat on both arms, so the override is complete by round 1 and it is not context length](../figures/fig4_q2.png)

**Q2: the override is complete after round one.** Margins toward the answer
the judge gave, at round 1 and at the full debate: steered 14.18 to 13.95, control
24.73 to 25.93. Flat on both arms, so the pre-registered length confound is ruled
out rather than assumed. The second round adds nothing the residual stream
registers. Steering wins the verdict but leaves the judge far less committed than an
honest debate.

**Limitations.** One judge, one dataset, instructed rather than emergent deception. The
primary test set is 23 debates. No causal intervention on the direction. An audit
found the corpus's true setup-leak rate is 3.8%, not the 0.6% my detector caught;
none is in the test set, and dropping it all moves Q0 by 0.004. The claim is
not that this judge has no truth representation, but that no linearly readable one
survives successful steering here.
