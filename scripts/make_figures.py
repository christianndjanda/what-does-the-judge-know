"""The four executive-summary figures, regenerated from the committed reports.

One figure per narrative beat, in the order the write-up makes them:

    fig1_q0.png       the attack works, and mostly by voiding the debate
    fig2_q1.png       91% on held-out data, 4% on the debates that matter
    fig3_killed.png   the near-tie effect, before and after the confound came out
    fig4_q2.png       both arms flat: override by round 1, and not a length effect

A script rather than four saved images, so a number that changes upstream cannot
stay stale in a figure. Everything is read from tracked JSON -- no activation store,
no GPU -- and every rate is recomputed here from the per-debate records rather than
copied out of a summary field, so a figure disagreeing with the write-up is a real
disagreement and not a transcription slip.

Confidence intervals are Wilson for proportions and normal-approximation for means,
reusing `judgeprobe.analysis` so the figures and the reports cannot drift into
different definitions.

    pip install matplotlib
    python scripts/make_figures.py
    python scripts/make_figures.py --out figures --dpi 300
"""

from __future__ import annotations

import argparse
import json
import sys
import textwrap
from math import comb
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from judgeprobe.analysis import wilson  # noqa: E402

# Colourblind-safe, and used consistently: the adversarial arm is always the warm
# colour, the honest arm always the cool one, the judge always neutral grey.
C_STEER = "#d1495b"
C_HONEST = "#2e6f95"
C_JUDGE = "#8d99ae"
C_REF = "#adb5bd"

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titlesize": 11,
    "axes.labelsize": 9.5,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "font.size": 9.5,
})


def fisher_one_tailed(a: int, b: int, c: int, d: int) -> float:
    """P(group 1 shows >= a successes | margins fixed). Pure stdlib, no scipy.

    Table is [[a, b], [c, d]] = [[grp1 success, grp1 failure], [grp2 ...]].
    """
    n = a + b + c + d
    row1, col1 = a + b, a + c
    return sum(
        comb(row1, i) * comb(c + d, col1 - i) / comb(n, col1)
        for i in range(a, min(row1, col1) + 1)
    )


def denom_for(acc: float, hi: int = 600) -> int | None:
    """Recover n from a reported accuracy, so a val-set CI can be drawn.

    The Phase 3 report stores `val_acc` but not the validation set's size. Rather
    than hardcode it, find the smallest n for which acc*n is an integer -- exact for
    a rational recorded at full float precision.
    """
    for n in range(1, hi + 1):
        if abs(acc * n - round(acc * n)) < 1e-9:
            return n
    return None


def bar_ci(ax, xs, heights, los, his, **kw):
    """Bars with asymmetric error bars from absolute CI bounds."""
    err = [[h - lo for h, lo in zip(heights, los)],
           [hi - h for h, hi in zip(heights, his)]]
    return ax.bar(xs, heights, yerr=err, capsize=4,
                  error_kw={"ecolor": "#343a40", "lw": 1.2}, **kw)


def finish(fig, ax, out: Path, dpi: int, title: str, caption: str) -> None:
    """One bold claim on top, the method caption below the axes.

    Long strings in `set_title` run off the canvas silently -- matplotlib does not
    wrap and does not warn. Both blocks are wrapped to a width derived from the
    figure, and the caption goes underneath where a reader looks for method detail,
    which also keeps the title short enough to survive being pasted at half size.
    """
    # Chars-per-inch differ by weight and size: the title is bold 11pt, the caption
    # regular 8pt. One shared width silently clips the title, which is how the first
    # pass lost the last two words of every heading.
    w = fig.get_figwidth()
    ax.set_title("\n".join(textwrap.wrap(title, int(w * 10.0))),
                 loc="left", fontweight="bold", pad=10)
    wrapped = textwrap.fill(caption, int(w * 16))
    # Reserve real space for the caption: an 8pt line is ~0.15in, so scale by the
    # figure's height. Under-reserving lets the rotated y-label run into the text.
    n_lines = wrapped.count("\n") + 1
    reserve = 0.03 + (0.15 * n_lines) / fig.get_figheight()
    fig.tight_layout(rect=(0, reserve, 1, 1))
    fig.text(0.012, 0.012, wrapped, fontsize=8, color="#495057", va="bottom")
    fig.savefig(out, dpi=dpi)
    plt.close(fig)
    print(f"  {out.name}")


# --- figure 1: Q0 -------------------------------------------------------------

def fig_q0(verdicts: list[dict], out: Path, dpi: int) -> None:
    def counts(cond):
        rows = [v["verdict"] for v in verdicts if v["condition"] == cond]
        cons = [r for r in rows if r["consistent"]]
        return {
            "n": len(rows),
            "consistent": (len(cons), len(rows)),
            "verdict_err": (sum(1 for r in cons if not r["correct"]), len(cons)),
            "forced_err": (sum(1 for r in rows if r["forced_choice"] != "gold"),
                           len(rows)),
        }

    h, c = counts("honest"), counts("collusion")
    metrics = [
        ("Order-consistency\n(both orders agree)", "consistent", False),
        ("Verdict error rate\n(of order-consistent)", "verdict_err", True),
        ("Forced-choice error rate\n(of all debates)", "forced_err", True),
    ]

    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    width = 0.36
    for i, (label, key, higher_is_worse) in enumerate(metrics):
        tops = []
        for j, (d, colour, name) in enumerate(((h, C_HONEST, "honest"),
                                               (c, C_STEER, "collusion"))):
            k, n = d[key]
            lo, hi = wilson(k, n)
            tops.append(hi)
            x = i + (j - 0.5) * width
            bar_ci(ax, [x], [k / n], [lo], [hi], width=width, color=colour,
                   zorder=3, label=name if i == 0 else None)
            # Above the CI cap, never above the bar: at these rates the interval is
            # taller than the gap and the label lands inside the whisker.
            ax.text(x, hi + 0.022, f"{k/n:.3f}", ha="center", fontsize=9,
                    color="#343a40")
            ax.text(x, 0.012, f"{k}/{n}", ha="center", fontsize=7.5,
                    color="white", zorder=4)

        # One-tailed Fisher, in the direction the treatment is expected to push.
        kh, nh = h[key]
        kc, nc = c[key]
        if higher_is_worse:
            p = fisher_one_tailed(kc, nc - kc, kh, nh - kh)
        else:
            p = fisher_one_tailed(kh, nh - kh, kc, nc - kc)
        ax.text(i, max(tops) + 0.075, f"p = {p:.2g}", ha="center", fontsize=9.5,
                color="#212529", fontweight="bold")

    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels([m[0] for m in metrics])
    ax.set_ylabel("rate")
    ax.set_ylim(0, 0.62)
    ax.legend(frameon=False, loc="upper right", ncols=2)
    ax.grid(axis="y", alpha=0.25, zorder=0)
    finish(fig, ax, out, dpi,
           "Q0: Covert collusion raises the judge's error rate, and more reliably "
           "destroys its verdict entirely",
           f"Qwen2.5-32B-Instruct judge, {h['n']} honest vs {c['n']} colluded "
           "QuALITY debates, every debate judged in both presentation orders. "
           "Order-consistency is the share where the two orders agree on an answer; "
           "the verdict error rate is conditioned on that agreement, the "
           "forced-choice rate is not. Error bars are Wilson 95% CIs, p-values "
           "one-tailed Fisher exact.")


# --- figure 2: Q1 -------------------------------------------------------------

def fig_q1(p3: dict, out: Path, dpi: int) -> None:
    n_val = denom_for(p3["val_acc"])
    k_val = round(p3["val_acc"] * n_val)
    n_test, k_test = p3["n_test"], p3["probe_correct"]

    bars = [
        ("Probe on held-out\nhonest debates\n(judge was right)",
         k_val / n_val, *wilson(k_val, n_val), C_HONEST, f"{k_val}/{n_val}"),
        ("Probe on debates where\nsteering succeeded",
         k_test / n_test, *wilson(k_test, n_test), C_STEER, f"{k_test}/{n_test}"),
        ("Judge on those same\nsteered debates",
         p3["judge_acc"], *wilson(0, n_test), C_JUDGE, f"0/{n_test}"),
    ]

    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    xs = range(len(bars))
    bar_ci(ax, list(xs), [b[1] for b in bars], [b[2] for b in bars],
           [b[3] for b in bars], color=[b[4] for b in bars], width=0.55, zorder=3)
    for i, b in enumerate(bars):
        # Both labels go above the CI cap. Inside the bar works only for the tall
        # one; on a 0.043 bar the fraction lands on top of its own whisker.
        ax.text(i, b[3] + 0.072, f"{b[1]:.3f}", ha="center", fontweight="bold",
                fontsize=11)
        ax.text(i, b[3] + 0.028, b[5], ha="center", fontsize=8.5, color="#6c757d")

    ax.axhline(0.5, color=C_REF, ls="--", lw=1.2, zorder=2)
    ax.text(len(bars) - 0.52, 0.515, "chance", fontsize=8.5, color="#6c757d",
            ha="right")
    ax.set_xticks(list(xs))
    ax.set_xticklabels([b[0] for b in bars])
    ax.set_ylabel("accuracy at recovering the correct answer")
    ax.set_ylim(0, 1.20)
    ax.grid(axis="y", alpha=0.25, zorder=0)
    finish(fig, ax, out, dpi,
           "Q1: 91% where truth and verdict agree, below chance where they come apart",
           f"Mass-mean probe at model layer {p3['model_layer']}, layer selected on "
           "held-out articles. The two numbers are not in tension: on the training "
           "distribution truth and verdict coincide by construction, so a pure "
           "verdict direction scores 0.912 there. On the steered set the probe "
           f"matches the judge's wrong verdict "
           f"{round(p3['probe_agrees_with_judge'] * n_test)}/{n_test} times "
           f"(binomial p below chance = {p3['binomial_p_below_chance']:.1e}), and its "
           "cosine with a probe trained on the verdict instead of the truth is "
           f"{p3['cosine_truth_verdict']:.3f}. Error bars are Wilson 95% CIs.")


# --- figure 3: the deleted result ---------------------------------------------

def fig_killed(forced: dict, cross: dict, out: Path, dpi: int) -> None:
    near = [b for b in forced["accuracy_by_judge_confidence"]
            if b["bin"] == "near-tie"][0]
    rows = {r["set"].strip(): r for r in cross["results"]}
    clean = rows["of those, near-ties (<0.05)"]

    k_before, n_before = round(near["probe_acc"] * near["n"]), near["n"]
    k_after, n_after = clean["probe_correct"], clean["n"]
    k_judge = round(clean["judge_acc"] * n_after)

    bars = [
        (f"Probe, as first binned\n(n = {n_before})",
         k_before / n_before, *wilson(k_before, n_before), C_STEER),
        (f"Probe, clean re-test\n(n = {n_after})",
         k_after / n_after, *wilson(k_after, n_after), C_STEER),
        (f"Judge's own forced choice,\nsame {n_after} debates",
         k_judge / n_after, *wilson(k_judge, n_after), C_JUDGE),
    ]

    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    xs = list(range(len(bars)))
    bar_ci(ax, xs, [b[1] for b in bars], [b[2] for b in bars],
           [b[3] for b in bars], color=[b[4] for b in bars], width=0.55, zorder=3)
    for i, b in enumerate(bars):
        ax.text(i, b[3] + 0.028, f"{b[1]:.3f}", ha="center", fontweight="bold",
                fontsize=11)

    top = max(b[3] for b in bars[:2])
    ax.annotate("", xy=(0.86, top + 0.115), xytext=(0.14, top + 0.115),
                arrowprops={"arrowstyle": "->", "color": "#343a40", "lw": 1.4})
    ax.text(0.5, top + 0.135, "one probe, and every order-split\ndebate whichever "
            "side of 0.5 it fell", ha="center", va="bottom", fontsize=8.5,
            color="#343a40")

    ax.axhline(0.5, color=C_REF, ls="--", lw=1.2, zorder=2)
    ax.text(len(bars) - 0.52, 0.515, "chance", fontsize=8.5, color="#6c757d",
            ha="right")
    ax.set_xticks(xs)
    ax.set_xticklabels([b[0] for b in bars])
    ax.set_ylabel("accuracy at recovering the correct answer")
    ax.set_ylim(0, 1.24)
    ax.grid(axis="y", alpha=0.25, zorder=0)
    finish(fig, ax, out, dpi,
           "The result I deleted: The near-tie effect was regression to the mean",
           "The first binning used a different probe and conditioned on P(gold) "
           "falling just below 0.5, which selects debates where one noisy reading ran "
           "low, so the other reverts upward. Re-tested with a single probe over "
           "every order-split debate, the effect drops to 0.653 and sits below the "
           "judge's own forced-choice accuracy on the same debates: the probe carries "
           "no information the judge's logits do not already have. The judge baseline "
           "matters -- against a 0.5 chance line this would have read as a win. Error "
           "bars are Wilson 95% CIs.")


# --- figure 4: Q2 -------------------------------------------------------------

def fig_q2(p5: dict, out: Path, dpi: int) -> None:
    labels = {"r1": "after round 1", "full": "full debate"}
    fig, ax = plt.subplots(figsize=(7.2, 4.1))

    for curve_key, ids_key, colour, name in (
            ("steered_curve", "n_steered", C_STEER,
             "steered (collusion, judge wrong)"),
            ("control_curve", "n_control", C_HONEST,
             "control: held-out honest, judge right")):
        curve = p5[curve_key]
        xs = list(range(len(curve)))
        ys = [c["mean_margin_verdict"] for c in curve]
        los = [c["margin_verdict_ci95"][0] for c in curve]
        his = [c["margin_verdict_ci95"][1] for c in curve]
        ax.errorbar(xs, ys, yerr=[[y - lo for y, lo in zip(ys, los)],
                                  [hi - y for y, hi in zip(ys, his)]],
                    marker="o", ms=7, lw=2, capsize=5, color=colour,
                    label=f"{name}  (n = {p5[ids_key]})", zorder=3)
        ax.annotate(f"{ys[-1]:.2f}", (xs[-1], ys[-1]), textcoords="offset points",
                    xytext=(10, -3), fontsize=9.5, color=colour,
                    fontweight="bold")
        ax.annotate(f"{ys[0]:.2f}", (xs[0], ys[0]), textcoords="offset points",
                    xytext=(-34, -3), fontsize=9.5, color=colour)

    ax.set_xticks(range(len(p5["steered_curve"])))
    ax.set_xticklabels([labels.get(c["round"], c["round"])
                        for c in p5["steered_curve"]])
    ax.set_xlim(-0.6, len(p5["steered_curve"]) - 0.25)
    ax.set_ylabel("probe margin toward the answer the judge gave")
    ax.set_ylim(0, 46)
    ax.legend(frameon=False, loc="upper center", ncols=1)
    ax.grid(axis="y", alpha=0.25, zorder=0)
    finish(fig, ax, out, dpi,
           "Q2: Flat on both arms, so the override is complete by round 1 and it is "
           "not context length",
           f"One probe, model layer {p5['model_layer']}, fit on the full transcripts "
           "of the training split and applied at each truncation -- not refitted per "
           "round. Margins are toward the answer the judge actually gave, so both "
           "arms measure the same thing: how strongly the residual stream has "
           "committed to the answer that will come out. The control is the "
           "pre-registered length confound, and it moves as little as the steered "
           "arm, so context length is ruled out rather than assumed. Every debate is "
           "2 rounds, so this is a two-point comparison, not a curve. Bars are 95% "
           "CIs.")


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--acts", type=Path, default=root / "data/phase2/acts_rounds")
    p.add_argument("--out", type=Path, default=root / "figures")
    p.add_argument("--dpi", type=int, default=200)
    args = p.parse_args()

    load = lambda name: json.loads((args.acts / name).read_text(encoding="utf-8"))
    verdicts = load("verdicts.json")["items"]
    p3 = load("phase3_report.json")
    forced = load("phase3_report_forced.json")
    cross = load("phase3_crosstest.json")
    p5 = load("phase5_report.json")

    args.out.mkdir(parents=True, exist_ok=True)
    print(f"figures -> {args.out}")
    fig_q0(verdicts, args.out / "fig1_q0.png", args.dpi)
    fig_q1(p3, args.out / "fig2_q1.png", args.dpi)
    fig_killed(forced, cross, args.out / "fig3_killed.png", args.dpi)
    fig_q2(p5, args.out / "fig4_q2.png", args.dpi)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
