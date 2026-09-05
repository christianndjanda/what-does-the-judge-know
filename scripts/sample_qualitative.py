"""Randomly sample debates for the write-up's qualitative-examples section.

The admissions doc asks for *randomly selected* raw examples placed just after the
executive summary -- "if bad data would sink your project, show me the data". This
project's headline rests on two things a reader cannot check from a number: that the
colluded debates are real arguments rather than degenerate text, and that the judge's
failures are genuine persuasion rather than parse noise. So the sampling is done here,
from a fixed seed, over an explicitly reported frame -- not by eye.

Three strata, one example each by default:

* `honest-correct`    -- the Phase 5 control arm: held-out honest debates the judge
                         got right. Shows what the baseline looks like.
* `collusion-steered` -- the Phase 3 test set: covert collusion, order-consistent,
                         judge wrong. The debates the whole result is about.
* `order-split`       -- the two presentation orders disagreed, so layout decided the
                         output. The failure mode Q0 found is *larger* than flipping.

Reads only committed JSON: no GPU, no activation store. Probe margins are attached
where `phase5_report.json` recorded them (the steered and control arms); the
order-split stratum has none, because no phase scored those debates -- reported as
absent rather than silently omitted.

    python scripts/sample_qualitative.py
    python scripts/sample_qualitative.py --seed 7 -n 2 --full-prompt
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from judgeprobe.debate import find_leak  # noqa: E402
from judgeprobe.judge import build_judge_messages, option_layout  # noqa: E402
from judgeprobe.transcripts import Transcript  # noqa: E402

STRATA = ("honest-correct", "collusion-steered", "order-split")

# `debate.find_leak` gates corpus generation, and its `\bi (?:was|am) (?:told|assigned)\b`
# needs a literal space -- so it passes "I'm assigned to argue X" and "my assigned
# answer", which are the forms the debaters actually produced. Found by the very
# first draw of this sampler, which is what random sampling is for. Audited here
# rather than fixed in `debate.py`: changing the generation-time gate now would
# re-quarantine a corpus whose results are already committed, so the corpus keeps
# its published composition and every example carries the annotation instead.
#
# Deliberately narrower than a guess at intent: "I am arguing for X" is an ordinary
# debate move, not a leak, and is not matched.
_AUDIT_LEAK_RE = re.compile(
    r"(?:\bi(?:'m|'ve| am| was| have been)\s+(?:told|assigned|instructed|tasked)\b"
    r"|\bmy assigned\s+(?:position|side|answer|argument|view)\b"
    r"|\bassigned to argue\b"
    r"|\bi(?:'m| am)\s+(?:meant|required)\s+to\b)",
    re.I,
)


def audit_leaks(t: Transcript) -> list[str]:
    """Setup-revealing phrases in a transcript, from either detector."""
    found = []
    for turn in t.turns:
        for m in _AUDIT_LEAK_RE.finditer(turn.text):
            found.append(m.group(0))
        hit = find_leak(turn.text)
        if hit:
            found.append(hit)
    return found


def load_transcripts(path: Path) -> dict[str, Transcript]:
    out = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                t = Transcript.from_dict(json.loads(line))
                out[t.transcript_id] = t
    return out


def load_verdicts(path: Path) -> dict[str, dict]:
    items = json.loads(path.read_text(encoding="utf-8"))["items"]
    return {it["transcript_id"]: it for it in items}


def load_margins(path: Path) -> tuple[dict[str, dict[str, float]], int | None, list[str]]:
    """Per-debate probe margins from a Phase 5 report, keyed id -> {round: margin}.

    `curve()` writes `margins_gold` aligned to the debates that actually had the round
    key in the store, which is a subset of the report's top-level id list. It happens
    to be the full list in the committed reports, but that is a property of the run,
    not a guarantee -- so the alignment is checked here and the whole round is dropped
    if it does not hold. Attaching a margin to the wrong transcript is exactly the
    kind of silent corruption these examples exist to rule out.
    """
    notes: list[str] = []
    if not path.exists():
        return {}, None, [f"{path.name} not found -- no probe margins attached"]

    rep = json.loads(path.read_text(encoding="utf-8"))
    layer = rep.get("model_layer")
    margins: dict[str, dict[str, float]] = {}
    for arm, ids_key, curve_key in (("steered", "steered_ids", "steered_curve"),
                                    ("control", "control_ids", "control_curve")):
        ids = rep.get(ids_key) or []
        for entry in rep.get(curve_key) or []:
            vals = entry.get("margins_gold") or []
            if len(vals) != len(ids) or entry.get("n") != len(ids):
                notes.append(
                    f"{arm}/{entry.get('round')}: {len(vals)} margins against "
                    f"{len(ids)} ids -- alignment unverifiable, round skipped")
                continue
            for i, v in zip(ids, vals):
                margins.setdefault(i, {})[entry["round"]] = v
    return margins, layer, notes


def frames(verdicts: dict[str, dict], phase5: Path) -> tuple[dict[str, list[str]], list[str]]:
    """The three sampling frames, plus any notes about how they were derived."""
    notes: list[str] = []
    steered: list[str] = []
    control: list[str] = []
    if phase5.exists():
        rep = json.loads(phase5.read_text(encoding="utf-8"))
        steered = list(rep.get("steered_ids") or [])
        control = list(rep.get("control_ids") or [])
    else:
        notes.append(f"{phase5.name} not found -- steered/control frames unavailable")

    order_split = [i for i, it in sorted(verdicts.items())
                   if not it["verdict"]["consistent"]]
    return {
        "honest-correct": sorted(control),
        "collusion-steered": sorted(steered),
        "order-split": order_split,
    }, notes


def fmt_example(tid: str, t: Transcript, item: dict, margins: dict, layer,
                *, full_prompt: bool) -> str:
    v = item["verdict"]
    layout = option_layout(t)  # forward order, exactly as the judge first saw it
    gold_letter = layout.gold_letter

    L = [f"### `{tid}`", ""]
    L.append(f"- **Condition:** {t.condition}  |  **Rounds:** {t.n_rounds}  "
             f"|  **Debaters:** {t.meta.get('debater_model', 'n/a')}")
    L.append(f"- **Article:** {t.meta.get('title', 'n/a')} "
             f"(`{t.meta.get('article_id', 'n/a')}`)  |  "
             f"**Gold speaks first:** {t.meta.get('gold_speaks_first', 'n/a')}")
    L.append("")
    L.append(f"**Question.** {t.question}")
    L.append("")
    L.append("Options as presented to the judge in the forward order "
             "(the reverse pass swaps them):")
    L.append("")
    L.append(f"- **A.** {layout.option_a}"
             + ("  <-- gold" if gold_letter == "A" else "  (best distractor)"))
    L.append(f"- **B.** {layout.option_b}"
             + ("  <-- gold" if gold_letter == "B" else "  (best distractor)"))
    L.append("")

    leaks = audit_leaks(t)
    if leaks:
        L.append(f"> **Setup leak in this debate:** {', '.join(repr(x) for x in leaks)}. "
                 "A debater referred to its own assignment, which the judge could in "
                 "principle read as a cue. Passed the committed `find_leak` gate; see "
                 "the corpus-level count above.")
        L.append("")

    L.append("**Transcript.** The judge sees only this and the options -- never the "
             "passage, and never the `gold`/`distractor` side labels below.")
    L.append("")
    for turn in t.turns:
        L.append(f"> **Round {turn.round_index} - {turn.speaker}** "
                 f"*(arguing for: {turn.side})*")
        L.append(">")
        for para in turn.text.split("\n"):
            L.append("> " + para if para.strip() else ">")
        L.append("")

    fwd, rev = v["forward"], v["reverse"]
    L.append("**Verdict.**")
    L.append("")
    L.append("| pass | generated | P(gold) renorm. over {A,B} |")
    L.append("| --- | --- | --- |")
    L.append(f"| forward | `{fwd['raw_output']}` | {fwd['p_gold']:.3f} |")
    L.append(f"| reverse | `{rev['raw_output']}` | {rev['p_gold']:.3f} |")
    L.append("")
    verdict_line = (f"- Order-averaged P(gold) **{v['p_gold']:.3f}**  |  "
                    f"order-consistent: **{v['consistent']}**")
    if v["consistent"]:
        verdict_line += f"  |  judge chose **{v['choice']}** (correct: {v['correct']})"
    else:
        verdict_line += "  |  no scored verdict: presentation order decided the output"
    L.append(verdict_line)
    L.append(f"- Forced choice (sign of the order-averaged P(gold)): "
             f"**{v['forced_choice']}**")
    L.append("")

    m = margins.get(tid)
    L.append("**Probe.**")
    L.append("")
    if m:
        L.append(f"Mass-mean probe at model layer {layer}, margin toward the gold "
                 f"answer (negative = the probe points at the distractor):")
        L.append("")
        for rk in sorted(m, key=lambda k: (k != "r1", k)):
            pick = "gold" if m[rk] > 0 else "distractor"
            L.append(f"- `{rk}`: **{m[rk]:+.2f}** -> probe reads **{pick}**")
    else:
        L.append("_No probe margin recorded for this debate: Phase 5 scored only the "
                 "steered and control arms, and this debate is in neither._")
    L.append("")

    if full_prompt:
        msgs = build_judge_messages(t, layout)
        L.append("<details><summary>Literal judge prompt (forward order)</summary>")
        L.append("")
        for msg in msgs:
            L.append(f"**[{msg['role']}]**")
            L.append("")
            L.append("```")
            L.append(msg["content"])
            L.append("```")
            L.append("")
        L.append("</details>")
        L.append("")
    return "\n".join(L)


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--transcripts", type=Path,
                   default=root / "data/phase2/transcripts.jsonl")
    p.add_argument("--acts", type=Path, default=root / "data/phase2/acts_rounds",
                   help="directory holding verdicts.json and phase5_report.json")
    p.add_argument("--seed", type=int, default=0,
                   help="fixed and reported in the output, so the draw is checkable")
    p.add_argument("-n", type=int, default=1, help="examples per stratum")
    p.add_argument("--full-prompt", action="store_true",
                   help="include the literal judge prompt for each example")
    p.add_argument("--out", type=Path,
                   default=root / "data/phase2/qualitative_examples.md")
    args = p.parse_args()

    transcripts = load_transcripts(args.transcripts)
    verdicts = load_verdicts(args.acts / "verdicts.json")
    phase5 = args.acts / "phase5_report.json"
    margins, layer, margin_notes = load_margins(phase5)
    frame_map, frame_notes = frames(verdicts, phase5)

    L = ["# Randomly selected qualitative examples", ""]
    L.append(f"Drawn by `scripts/sample_qualitative.py --seed {args.seed} -n {args.n}`. "
             "Uniform over each frame below -- **not** hand-picked. Rerunning with the "
             "same seed reproduces exactly these debates; changing the seed draws "
             "different ones, and this section was not regenerated to find a "
             "flattering draw.")
    L.append("")
    L.append(f"Judge: `{next(iter(verdicts.values()))['model_id']}`. "
             f"Corpus: {len(verdicts)} debates.")
    L.append("")
    L.append("| stratum | frame size | definition |")
    L.append("| --- | --- | --- |")
    L.append(f"| honest-correct | {len(frame_map['honest-correct'])} | "
             "held-out honest debates the judge got right (Phase 5 control arm) |")
    L.append(f"| collusion-steered | {len(frame_map['collusion-steered'])} | "
             "covert collusion, order-consistent, judge wrong (Phase 3 test set) |")
    L.append(f"| order-split | {len(frame_map['order-split'])} | "
             "the two presentation orders disagreed; no scored verdict |")
    L.append("")

    # Corpus-level leak audit, reported whether or not a sampled debate is affected,
    # so the rate cannot be inferred only from a lucky draw.
    leaked = {i: audit_leaks(t) for i, t in transcripts.items()}
    leaked = {i: v for i, v in leaked.items() if v}
    if leaked:
        by_cond: dict[str, int] = {}
        for i in leaked:
            c = transcripts[i].condition
            by_cond[c] = by_cond.get(c, 0) + 1
        conds = ", ".join(f"{v} {k}" for k, v in sorted(by_cond.items()))
        L.append(f"**Setup-leak audit.** {len(leaked)} of {len(transcripts)} debates "
                 f"({len(leaked) / len(transcripts):.1%}; {conds}) contain a phrase in "
                 "which a debater refers to its own assignment. These passed the "
                 "generation-time `find_leak` gate, whose pattern misses contracted and "
                 "possessive forms; they are audited here rather than removed, so the "
                 "corpus keeps the composition the committed results were computed on.")
        for key, name in (("honest-correct", "Phase 5 control arm"),
                          ("collusion-steered", "Phase 3 steered test set")):
            overlap = sorted(set(frame_map[key]) & set(leaked))
            L.append(f"- In the {name} ({len(frame_map[key])} debates): "
                     + (", ".join(f"`{i}`" for i in overlap) if overlap else "**none**"))
        L.append("")

    for note in frame_notes + margin_notes:
        L.append(f"> WARNING: {note}")
    if frame_notes or margin_notes:
        L.append("")

    for name in STRATA:
        frame = frame_map[name]
        L.append(f"## {name}")
        L.append("")
        if not frame:
            L.append("_Frame is empty -- nothing to sample._")
            L.append("")
            continue
        # Seeded per stratum, so adding a stratum or changing one frame does not
        # reshuffle the others' draws.
        rng = random.Random(f"{args.seed}:{name}")
        k = min(args.n, len(frame))
        if k < args.n:
            L.append(f"_Frame holds only {len(frame)}; sampling all of them._")
            L.append("")
        for tid in rng.sample(frame, k):
            t = transcripts.get(tid)
            if t is None:
                L.append(f"_`{tid}` sampled but missing from the transcript file._")
                L.append("")
                continue
            L.append(fmt_example(tid, t, verdicts[tid], margins, layer,
                                 full_prompt=args.full_prompt))

    text = "\n".join(L)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text, encoding="utf-8")
    print(text)
    print(f"\n[written to {args.out}]", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
