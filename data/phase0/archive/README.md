# Archived Phase 0 reports

`phase0_gates.py` overwrites `gate_report.json` on every invocation, so a later
partial run (`--gate c`) replaces an earlier full one. These are the reports that
would otherwise have been lost.

## gate_report_gold_always_first.json

The first full A/B/C run on Qwen2.5-32B, **before** `gold_speaks_first` was
balanced. All three gates passed: Gate C best-layer LOO 1.00 with 33/64 layers above
threshold.

That result was an artifact. Every transcript in that corpus had the gold side
speaking first, so "does the appended answer match Debater A" was perfectly
correlated with truth, and the probe was reading that rather than truth. Rerunning
with speaking order balanced dropped best LOO to 0.60 with zero layers above
threshold -- see `../gate_report.json`.

Kept because the before/after pair is the evidence for that finding, and the "before"
half cannot be regenerated without deliberately reintroducing the confound.

## Note on `../verdicts.json`

Gate A was only ever run on the pre-fix corpus, so `verdicts.json` describes those
transcripts -- not the balanced `transcripts.jsonl` now in the repo. The two do not
correspond. Re-run `--gate a` on the current corpus before using it for anything.
