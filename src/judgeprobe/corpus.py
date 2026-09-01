"""Phase 2 corpus runner: plan the debates, generate them in parallel, stream to disk.

The doc's warning is that serial API calls over long QuALITY contexts eat the whole
budget. A debate cannot be parallelised inside itself -- round 2 depends on round 1 --
so the parallelism is across debates, with each worker owning one debate end to end.

Same disk discipline as the activation store: append as you go, never hold the corpus
in memory, and let a rerun skip what is already written. An interrupted run is worth
money here, not just time.

Failures and leaks are quarantined into their own files rather than dropped, so the
rates are recoverable afterwards. A debate whose text gives away the setup is unusable
-- it tells the judge the answer -- but the rate at which the debaters leak is itself
worth reporting.
"""

from __future__ import annotations

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from .debate import (
    CONDITIONS, DebateConfig, Usage, classify_covertness, generate_debate,
)
from .quality import QualityQuestion
from .transcripts import (
    Transcript, assign_balanced_letters, balanced_flags, letter_balance,
    read_jsonl, speaker_balance, write_jsonl,
)


@dataclass
class DebateSpec:
    question: QualityQuestion
    condition: str
    gold_speaks_first: bool

    @property
    def transcript_id(self) -> str:
        return f"p2-{self.condition}-{self.question.question_id}"


def plan_corpus(questions: list[QualityQuestion], conditions=("honest", "collusion"),
                *, seed: int = 0, paired: bool = False) -> list[DebateSpec]:
    """Assign questions to conditions and balance the speaking order.

    By default the conditions get **disjoint** questions. Phase 3 trains on the honest
    transcripts and tests on the colluded ones, so a question appearing in both puts
    the same question text, the same two answer strings and the same gold answer on
    both sides of the split. A probe could then separate the test pairs by having
    memorised that question rather than by carrying a truth representation, and the
    headline number would be inflated in a way no downstream control would catch.

    `paired=True` puts every question in every condition, which is the right design
    for comparing Q0 within a question but not for training a probe. Phase 3 must not
    use a paired corpus without a question-level split.
    """
    for c in conditions:
        if c not in CONDITIONS:
            raise ValueError(f"unknown condition {c!r}")

    groups: dict[str, list[QualityQuestion]] = {}
    if paired:
        groups = {c: list(questions) for c in conditions}
    else:
        # Assign whole *articles*, not individual questions. With one question per
        # article the two are the same thing; with several, splitting an article
        # across conditions would put the same story in both the probe's training
        # and test sets, and a probe could separate the test pairs by recognising
        # the story rather than by carrying any truth representation.
        by_article: dict[str, list[QualityQuestion]] = {}
        for q in questions:
            by_article.setdefault(q.article_id, []).append(q)
        for i, article_id in enumerate(sorted(by_article)):
            groups.setdefault(conditions[i % len(conditions)], []).extend(
                by_article[article_id])

    specs: list[DebateSpec] = []
    for offset, condition in enumerate(conditions):
        group = groups.get(condition, [])
        # Exactly half of each condition has the gold side speaking first -- balanced,
        # not merely random, because at Phase 0 sizes "random" delivered 5/5.
        #
        # Balanced *within article* as well, for the reason the option letter had to
        # be: balancing a nuisance factor marginally leaves it free to correlate with
        # something else. If a whole article's debates all had gold speaking first,
        # speaking order would track story identity, and story identity is exactly
        # what the article-level split exists to keep out of the probe.
        # The flag is a function of (article index, question index) only -- never of
        # how many questions were drawn. `balanced_flags(k)` would shuffle differently
        # at k=2 and k=4, silently changing the speaking order of debates that already
        # exist and forcing them to be regenerated. Strict alternation within an
        # article, with the starting value alternating across articles, is exactly
        # balanced for even k, balanced overall for odd k, and stable as k grows.
        per_article: dict[str, list[QualityQuestion]] = {}
        for q in group:
            per_article.setdefault(q.article_id, []).append(q)
        for j, article_id in enumerate(sorted(per_article)):
            qs = sorted(per_article[article_id], key=lambda q: q.question_id)
            specs += [DebateSpec(q, condition, (i + j + offset) % 2 == 0)
                      for i, q in enumerate(qs)]
    return specs


class CorpusWriter:
    """Append-only jsonl for transcripts, leaks and failures, safe across threads."""

    def __init__(self, out_dir: Path):
        self.dir = Path(out_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.transcripts = self.dir / "transcripts.jsonl"
        self.leaked = self.dir / "leaked.jsonl"
        self.failures = self.dir / "failures.jsonl"
        self._lock = threading.Lock()

    def done_ids(self, include_failures: bool = True) -> set[str]:
        """Ids already written -- to transcripts *or* quarantine, so reruns settle.

        `include_failures=False` re-attempts previously failed debates. Needed when
        the failures were caused by something global rather than by the debates
        themselves -- a bad key quarantines the entire corpus, and those ids would
        otherwise be skipped forever.
        """
        sources = [(self.transcripts, "transcript_id"), (self.leaked, "transcript_id")]
        if include_failures:
            sources.append((self.failures, "transcript_id"))
        ids: set[str] = set()
        for path, key in sources:
            if path.exists():
                with path.open(encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            ids.add(json.loads(line)[key])
        return ids

    def _append(self, path: Path, record: dict) -> None:
        with self._lock, path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def write_transcript(self, t: Transcript) -> None:
        self._append(self.transcripts, t.to_dict())

    def write_leaked(self, t: Transcript) -> None:
        self._append(self.leaked, t.to_dict())

    def write_failure(self, transcript_id: str, condition: str, error: str) -> None:
        self._append(self.failures, {"transcript_id": transcript_id,
                                     "condition": condition, "error": error})


def _run_one(client, spec: DebateSpec, cfg: DebateConfig):
    result = generate_debate(client, spec.question, condition=spec.condition,
                             gold_speaks_first=spec.gold_speaks_first, cfg=cfg)
    if result.transcript is not None and cfg.classify:
        try:
            result.transcript.meta["covertness"] = classify_covertness(
                client, result.transcript, model=cfg.model, usage=result.usage)
        except Exception as e:
            # A failed audit is not a failed debate: keep the transcript, record why
            # the covertness label is missing.
            result.transcript.meta["covertness"] = {"error": f"{type(e).__name__}: {e}"}
        result.transcript.meta["usage"] = result.usage.to_dict(cfg.model)
    return result


#: Abort a run after this many failures with no successes. A per-debate failure is
#: worth recording and moving on from -- that is what the quarantine is for. A
#: *global* failure (bad credentials, wrong model id, no network) fails every debate
#: identically, and because `done_ids` counts failures as done so reruns settle, an
#: unchecked run writes one quarantine entry per spec and permanently poisons the
#: resume state. That happened: a run with a placeholder API key wrote 527
#: AuthenticationErrors, after which the real run found "0 to go".
FAIL_FAST_THRESHOLD = 5


def build_corpus(client, specs: list[DebateSpec], writer: CorpusWriter, *,
                 cfg: DebateConfig, workers: int = 8, resume: bool = True,
                 retry_failed: bool = False, log=print) -> dict:
    """Generate every spec, in parallel, appending as results land."""
    done = writer.done_ids(include_failures=not retry_failed) if resume else set()
    todo = [s for s in specs if s.transcript_id not in done]
    if done:
        log(f"resuming: {len(done)} already written, {len(todo)} to go")

    total = Usage()
    counts = {"ok": 0, "leaked": 0, "failed": 0}
    covert = {"HONEST": 0, "THROWN": 0, "unknown": 0}
    started = time.time()
    aborted = None

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_run_one, client, s, cfg): s for s in todo}
        for i, fut in enumerate(as_completed(futures), 1):
            spec = futures[fut]
            try:
                result = fut.result()
            except Exception as e:  # a worker itself died; record and continue
                writer.write_failure(spec.transcript_id, spec.condition,
                                     f"{type(e).__name__}: {e}")
                counts["failed"] += 1
                continue

            total.merge(result.usage)
            if result.transcript is None:
                writer.write_failure(spec.transcript_id, spec.condition,
                                     result.error or "unknown")
                counts["failed"] += 1
                status = f"FAILED {result.error}"
            elif result.leak:
                writer.write_leaked(result.transcript)
                counts["leaked"] += 1
                status = f"LEAKED {result.leak!r}"
            else:
                writer.write_transcript(result.transcript)
                counts["ok"] += 1
                label = (result.transcript.meta.get("covertness") or {}).get("verdict")
                covert[label if label in covert else "unknown"] += 1
                status = f"ok  covertness={label or '-'}"

            elapsed = time.time() - started
            log(f"[{i}/{len(todo)}] {spec.transcript_id}  {status}  "
                f"${total.cost(cfg.model):.2f} so far  "
                f"eta {(elapsed / i * (len(todo) - i)) / 60:.1f}m")

            # Nothing has succeeded and failures are piling up: this is a global
            # fault, not a run of unlucky debates. Stop before quarantining the
            # whole corpus.
            if counts["ok"] == 0 and counts["failed"] >= FAIL_FAST_THRESHOLD:
                aborted = (f"{counts['failed']} failures, 0 successes -- stopping. "
                           f"This looks like a global fault (credentials, model id, "
                           f"network) rather than bad debates. Fix it, then rerun "
                           f"with --retry-failed to clear the quarantined ids.")
                log(f"\nABORTED: {aborted}")
                for f in futures:
                    f.cancel()
                break

    return {
        "model": cfg.model,
        "aborted": aborted,
        "n_planned": len(specs),
        "n_attempted": len(todo),
        "counts": counts,
        "covertness": covert,
        "covert_rate": (covert["HONEST"] / counts["ok"]) if counts["ok"] else None,
        "elapsed_s": round(time.time() - started, 1),
        "usage": total.to_dict(cfg.model),
    }


def finalise(writer: CorpusWriter, *, seed: int = 0) -> dict:
    """Assign balanced option letters over the finished corpus and rewrite it.

    The letter is the one nuisance factor that can be fixed after generation, since
    it lives in the judge's prompt rather than the transcript. Doing it here, over
    the whole corpus at once, is what makes it exactly balanced within condition
    rather than balanced in expectation -- and a resumed run gets the same treatment
    as a single-shot one.
    """
    if not writer.transcripts.exists():
        return {"n": 0}
    transcripts = read_jsonl(writer.transcripts)
    assign_balanced_letters(transcripts, seed=seed)
    write_jsonl(transcripts, writer.transcripts)
    return {
        "n": len(transcripts),
        "letter_balance": letter_balance(transcripts),
        "speaker_balance": speaker_balance(transcripts),
    }
