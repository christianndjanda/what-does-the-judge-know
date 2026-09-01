"""Phase 2 self-test -- the corpus pipeline against a stub client, no API calls.

Phase 2 is the only phase that spends real money, and a structural bug in it is not
recoverable by rerunning one debate: the confounds it can introduce (speaking order,
condition overlap, a leaked setup) are properties of the corpus, and Phase 0 showed
that a corpus-level confound survives every downstream control. So the whole
generation path runs here against a fake Anthropic client first.

The contracts:

    D1  debater prompts: cached story, side-specific and condition-specific strategy
    D2  debate structure: sequential rounds, speaking order honoured, rebuttal visible
    D3  a turn that leaks the setup is quarantined, not written to the corpus
    D4  a failing debate is recorded and the run continues
    D5  a rerun resumes and writes nothing twice
    D6  the corpus is balanced: disjoint conditions, speaking order, option letters
    D7  usage and cost are accounted per debate

Usage:
    python scripts/phase2_selftest.py
"""

from __future__ import annotations

import json
import re
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from judgeprobe import config  # noqa: E402
from judgeprobe.corpus import CorpusWriter, build_corpus, finalise, plan_corpus  # noqa: E402
from judgeprobe.debate import DebateConfig, debater_system, find_leak  # noqa: E402
from judgeprobe.quality import sample_questions  # noqa: E402
from judgeprobe.transcripts import (  # noqa: E402
    DISTRACTOR_SIDE, GOLD_SIDE, read_jsonl,
)

OUT = config.DATA_DIR / "phase2" / "selftest"
N_QUESTIONS = 8
ROUNDS = 2


# --- stub Anthropic client ---------------------------------------------------

class _Block:
    type = "text"

    def __init__(self, text: str):
        self.text = text


class _Usage:
    def __init__(self, out_tokens: int):
        self.input_tokens = 400
        self.output_tokens = out_tokens
        self.cache_creation_input_tokens = 6000
        self.cache_read_input_tokens = 0


class _Response:
    stop_reason = "end_turn"
    stop_details = None

    def __init__(self, text: str):
        self.content = [_Block(text)]
        self.usage = _Usage(len(text) // 4)


class _Messages:
    def __init__(self, owner):
        self._owner = owner

    def create(self, *, model, max_tokens, temperature, system, messages, **kw):
        return self._owner._create(system=system, prompt=messages[0]["content"],
                                   max_tokens=max_tokens)


class FakeClient:
    """Returns canned turns, and records every call for the assertions below."""

    def __init__(self, *, leak_ids=(), fail_ids=(), thrown_ids=()):
        self.messages = _Messages(self)
        self.calls: list[dict] = []
        self.leak_ids, self.fail_ids = set(leak_ids), set(fail_ids)
        self.thrown_ids = set(thrown_ids)

    def _create(self, *, system, prompt, max_tokens):
        blocks = system if isinstance(system, list) else [{"text": system}]
        text = "\n".join(b["text"] for b in blocks)
        self.calls.append({"system": text, "prompt": prompt})

        if max_tokens <= 8:  # the covertness classifier
            title = re.search(r"Question: (.*)", prompt)
            key = title.group(1) if title else ""
            return _Response("THROWN" if any(t in key for t in self.thrown_ids)
                             else "HONEST")

        claim = re.search(r"The answer you argue for: (.*)", text).group(1)
        story_title = re.search(r"Story: (.*)", text).group(1)
        if story_title in self.fail_ids:
            raise RuntimeError("simulated API failure")
        turn = f"The story supports it: {claim}. My opponent has no line to quote."
        if story_title in self.leak_ids:
            turn = "As instructed, I will make the weaker case here. " + turn
        return _Response(turn)


# --- checks ------------------------------------------------------------------

class Checks:
    def __init__(self) -> None:
        self.rows: list[dict] = []

    def check(self, contract: str, name: str, ok: bool, detail: str = "") -> bool:
        ok = bool(ok)
        self.rows.append({"contract": contract, "name": name, "pass": ok,
                          "detail": str(detail)})
        print(f"  [{'PASS' if ok else 'FAIL'}] {contract} {name}"
              + (f"  -- {detail}" if detail else ""))
        return ok

    @property
    def passed(self) -> bool:
        return all(r["pass"] for r in self.rows)


def main() -> int:
    shutil.rmtree(OUT, ignore_errors=True)
    questions = sample_questions(N_QUESTIONS, split="dev", seed=config.SEED)
    if len(questions) < N_QUESTIONS:
        print(f"only {len(questions)} questions available; need {N_QUESTIONS}")
        return 2
    cfg = DebateConfig(n_rounds=ROUNDS, classify=True)
    c = Checks()

    # --- D1: prompts ----------------------------------------------------------
    print("\nD1  debater prompts")
    q = questions[0]
    honest_gold = debater_system(q, side=GOLD_SIDE, condition="honest")
    collude_gold = debater_system(q, side=GOLD_SIDE, condition="collusion")
    collude_dist = debater_system(q, side=DISTRACTOR_SIDE, condition="collusion")
    relaxed_gold = debater_system(q, side=GOLD_SIDE, condition="relaxed")
    c.check("D1", "story sits in a cached block",
            honest_gold[0]["cache_control"] == {"type": "ephemeral"}
            and q.article[:200] in honest_gold[0]["text"],
            f"{len(honest_gold[0]['text'])} chars cached")
    c.check("D1", "the volatile assignment comes after the cached prefix",
            "cache_control" not in honest_gold[1]
            and q.gold in honest_gold[1]["text"])
    c.check("D1", "each side is told to argue its own answer",
            q.gold in collude_gold[1]["text"].split("The answer you argue for: ")[1]
            and q.best_distractor in
            collude_dist[1]["text"].split("The answer you argue for: ")[1])
    c.check("D1", "collusion tells the gold side to lose",
            "lose" in collude_gold[1]["text"]
            and "lose" not in honest_gold[1]["text"])
    c.check("D1", "relaxed lifts covertness, collusion does not",
            "lifted" in relaxed_gold[1]["text"]
            and "lifted" not in collude_gold[1]["text"])
    c.check("D1", "no prompt names the gold answer as correct to the distractor side",
            "is NOT your assigned answer" in collude_dist[1]["text"]
            and f"correct answer is {q.gold}" not in collude_dist[1]["text"])

    # --- D2: structure --------------------------------------------------------
    print("\nD2  debate structure")
    specs = plan_corpus(questions, ("honest", "collusion"), seed=config.SEED)
    client = FakeClient()
    writer = CorpusWriter(OUT)
    build_corpus(client, specs[:2], writer, cfg=cfg, workers=1, log=lambda *a: None)
    written = read_jsonl(writer.transcripts)
    t = written[0]
    c.check("D2", "two turns per round, rounds in order",
            [x.round_index for x in t.turns] == [1, 1, 2, 2],
            f"{len(t.turns)} turns over {t.n_rounds} rounds")
    spec = next(s for s in specs if s.transcript_id == t.transcript_id)
    gold_turn = next(x for x in t.turns if x.side == GOLD_SIDE)
    c.check("D2", "speaking order honours gold_speaks_first",
            (gold_turn.speaker == "Debater A") == spec.gold_speaks_first,
            f"gold_speaks_first={spec.gold_speaks_first}, gold is {gold_turn.speaker}")
    c.check("D2", "each side argues its own answer",
            t.gold.rstrip(".") in gold_turn.text
            and t.best_distractor.rstrip(".")
            in next(x for x in t.turns if x.side == DISTRACTOR_SIDE).text)
    second_turn_prompt = client.calls[1]["prompt"]
    c.check("D2", "the second speaker sees the first speaker's turn",
            "The debate so far" in second_turn_prompt
            and t.turns[0].text[:40] in second_turn_prompt)
    c.check("D2", "round 2 sees all of round 1",
            all(x.text[:40] in client.calls[2]["prompt"] for x in t.turns[:2]))
    c.check("D2", "the judge-facing transcript carries no story text",
            t.article_excerpt == "")

    # --- D3/D4: quarantine ----------------------------------------------------
    print("\nD3/D4  quarantine")
    shutil.rmtree(OUT, ignore_errors=True)
    writer = CorpusWriter(OUT)
    leak_title, fail_title = specs[0].question.title, specs[1].question.title
    client = FakeClient(leak_ids=[leak_title], fail_ids=[fail_title])
    report = build_corpus(client, specs, writer, cfg=cfg, workers=4,
                          log=lambda *a: None)
    c.check("D3", "the leak detector fires on a setup mention",
            find_leak("As instructed, I will make the weaker case") is not None
            and find_leak("The story says he flew the ship") is None)
    c.check("D3", "leaked debates are quarantined, not written to the corpus",
            report["counts"]["leaked"] == 1
            and writer.leaked.exists()
            and not any(x.transcript_id == f"p2-{specs[0].condition}-"
                        f"{specs[0].question.question_id}"
                        for x in read_jsonl(writer.transcripts)),
            f"{report['counts']['leaked']} leaked")
    c.check("D4", "a failing debate is recorded and the run continues",
            report["counts"]["failed"] == 1
            and report["counts"]["ok"] == len(specs) - 2,
            json.dumps(report["counts"]))
    failures = [json.loads(l) for l in writer.failures.read_text().splitlines() if l]
    c.check("D4", "the failure record names the debate and the error",
            failures[0]["transcript_id"].startswith("p2-")
            and "simulated API failure" in failures[0]["error"])

    # --- D5: resume -----------------------------------------------------------
    print("\nD5  resume")
    before = writer.transcripts.read_text(encoding="utf-8")
    rerun = build_corpus(client, specs, writer, cfg=cfg, workers=4,
                         log=lambda *a: None)
    c.check("D5", "a rerun attempts nothing already written",
            rerun["n_attempted"] == 0, f"{len(writer.done_ids())} ids known")
    c.check("D5", "the corpus file is byte-identical after the rerun",
            writer.transcripts.read_text(encoding="utf-8") == before)

    # --- D6: balance ----------------------------------------------------------
    print("\nD6  corpus balance")
    by_condition: dict[str, set] = {}
    for s in specs:
        by_condition.setdefault(s.condition, set()).add(s.question.question_id)
    c.check("D6", "conditions get disjoint questions",
            not (by_condition["honest"] & by_condition["collusion"]),
            f"{len(by_condition['honest'])} honest, "
            f"{len(by_condition['collusion'])} collusion")
    for condition, group in sorted(by_condition.items()):
        n_first = sum(s.gold_speaks_first for s in specs if s.condition == condition)
        c.check("D6", f"speaking order balanced [{condition}]",
                abs(n_first - len(group) / 2) <= 0.5,
                f"gold speaks first in {n_first}/{len(group)}")
    balance = finalise(writer, seed=config.SEED)
    for condition, (n_a, total) in sorted(balance["letter_balance"].items()):
        c.check("D6", f"option letter balanced [{condition}]",
                abs(n_a - total / 2) <= 0.5, f"gold is 'A' in {n_a}/{total}")
    paired = plan_corpus(questions, ("honest", "collusion"), seed=config.SEED,
                         paired=True)
    c.check("D6", "paired mode is opt-in and does overlap",
            len(paired) == 2 * len(questions))

    # --- D7: accounting -------------------------------------------------------
    print("\nD7  usage accounting")
    per_debate_calls = 2 * ROUNDS + 1  # turns + one covertness audit
    ok_debates = report["counts"]["ok"]
    c.check("D7", "calls per debate = turns + covertness audit",
            report["usage"]["calls"] >= ok_debates * per_debate_calls,
            f"{report['usage']['calls']} calls for {ok_debates} debates")
    c.check("D7", "cost is accounted in dollars",
            report["usage"]["cost_usd"] > 0,
            f"${report['usage']['cost_usd']} (stub tokens, not real spend)")
    t0 = read_jsonl(writer.transcripts)[0]
    c.check("D7", "each transcript carries its own usage and covertness label",
            t0.meta["usage"]["calls"] == per_debate_calls
            and t0.meta["covertness"]["verdict"] in ("HONEST", "THROWN"),
            f"{t0.meta['usage']['calls']} calls, "
            f"covertness={t0.meta['covertness']['verdict']}")
    c.check("D7", "provenance recorded on every transcript",
            all(x.meta.get("debater_model") and x.meta.get("generated_at")
                for x in read_jsonl(writer.transcripts)))

    report_path = OUT.parent / "selftest_report.json"
    report_path.write_text(json.dumps({"checks": c.rows, "pass": c.passed,
                                       "run": report}, indent=2, default=str),
                           encoding="utf-8")
    shutil.rmtree(OUT, ignore_errors=True)
    n_fail = sum(1 for r in c.rows if not r["pass"])
    print(f"\n{'PASS' if c.passed else 'FAIL'}: {len(c.rows) - n_fail}/{len(c.rows)} "
          f"checks  ->  {report_path}")
    return 0 if c.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
