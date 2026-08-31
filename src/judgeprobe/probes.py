"""Linear probes on cached residual streams.

Primary method is mass-mean (difference of class means), per Marks & Tegmark 2023
-- the design doc makes logistic regression a secondary comparison only. Both are
implemented here in numpy/torch rather than pulling in scikit-learn: one fewer
aarch64 wheel to worry about on the GH200, and these two fits are ten lines each.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ProbeResult:
    """A fitted direction at one layer, plus its threshold."""

    layer: int
    direction: np.ndarray  # (d_model,), unit norm
    threshold: float
    method: str

    def score(self, acts: np.ndarray) -> np.ndarray:
        """Signed projection onto the direction. Positive = class 1 (true)."""
        acts = np.atleast_2d(acts)
        return acts @ self.direction - self.threshold

    def predict(self, acts: np.ndarray) -> np.ndarray:
        return (self.score(acts) > 0).astype(int)


def _unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def fit_mass_mean(pos: np.ndarray, neg: np.ndarray, *, layer: int = -1,
                  iid: bool = False) -> ProbeResult:
    """Mass-mean probe from positive (true) and negative (false) activations.

    `pos`/`neg` are `(n, d_model)` at a single layer. With `iid=True` the direction
    is whitened by the pooled within-class covariance (Marks & Tegmark's "IID"
    variant); that needs n >> d_model to be stable, so it is off by default and
    should stay off at Phase 0 sample sizes.
    """
    mu_pos, mu_neg = pos.mean(0), neg.mean(0)
    direction = mu_pos - mu_neg
    if iid:
        centred = np.concatenate([pos - mu_pos, neg - mu_neg], axis=0)
        cov = np.cov(centred, rowvar=False) + 1e-3 * np.eye(centred.shape[1])
        direction = np.linalg.solve(cov, direction)
    direction = _unit(direction)
    # Midpoint of the projected class means: the natural threshold when the two
    # classes are equally sized, which they are by construction (contrast pairs).
    threshold = float((mu_pos @ direction + mu_neg @ direction) / 2)
    return ProbeResult(layer=layer, direction=direction, threshold=threshold,
                       method="mass_mean_iid" if iid else "mass_mean")


def fit_logistic(pos: np.ndarray, neg: np.ndarray, *, layer: int = -1,
                 l2: float = 1.0, steps: int = 300, lr: float = 0.05) -> ProbeResult:
    """Secondary comparison only (design doc s0.5). Standardised inputs, L2, LBFGS."""
    import torch

    X = np.concatenate([pos, neg], axis=0)
    y = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Xn = torch.tensor((X - mu) / sd, dtype=torch.float32)
    yt = torch.tensor(y, dtype=torch.float32)

    w = torch.zeros(Xn.shape[1], requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([w, b], lr=lr, max_iter=steps)

    def closure():
        opt.zero_grad()
        logits = Xn @ w + b
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, yt)
        loss = loss + l2 * w.pow(2).mean()
        loss.backward()
        return loss

    opt.step(closure)
    # Undo standardisation so the direction lives in raw activation space.
    w_raw = (w.detach().numpy() / sd)
    b_raw = float(b.detach().numpy()[0]) - float(w_raw @ mu)
    norm = np.linalg.norm(w_raw)
    if norm > 0:
        b_raw /= norm
    return ProbeResult(layer=layer, direction=_unit(w_raw), threshold=-b_raw,
                       method="logistic")


def accuracy(probe: ProbeResult, pos: np.ndarray, neg: np.ndarray) -> float:
    correct = int(probe.predict(pos).sum()) + int((1 - probe.predict(neg)).sum())
    return correct / (len(pos) + len(neg))


def loo_accuracy(pos: np.ndarray, neg: np.ndarray, *, fit=fit_mass_mean, **kw) -> float:
    """Leave-one-pair-out accuracy at a single layer.

    Contrast pairs are matched, so a held-out example's partner must be held out
    too -- otherwise the pair's shared context leaks into the training means. With
    ten Phase 0 examples, in-sample accuracy is meaningless; this is the number
    Gate C should be read off.
    """
    n = min(len(pos), len(neg))
    if n < 2:
        raise ValueError("need at least 2 pairs for leave-one-out")
    correct = 0
    for i in range(n):
        keep = [j for j in range(n) if j != i]
        probe = fit(pos[keep], neg[keep], **kw)
        correct += int(probe.predict(pos[i]).item() == 1)
        correct += int(probe.predict(neg[i]).item() == 0)
    return correct / (2 * n)


def sweep_layers(pos: np.ndarray, neg: np.ndarray, *, fit=fit_mass_mean,
                 loo: bool = True, **kw) -> list[dict]:
    """Fit one probe per layer.

    `pos`/`neg` are `(n, n_layers, d_model)`. Returns per-layer accuracy, sorted
    by layer, with both the in-sample and (by default) leave-one-out figures.
    """
    if pos.ndim != 3 or neg.ndim != 3:
        raise ValueError(f"expected (n, n_layers, d_model), got {pos.shape} / {neg.shape}")
    rows = []
    for layer in range(pos.shape[1]):
        p, q = pos[:, layer, :], neg[:, layer, :]
        probe = fit(p, q, layer=layer, **kw)
        row = {
            "layer": layer,
            "method": probe.method,
            "train_acc": accuracy(probe, p, q),
            "mean_diff_norm": float(np.linalg.norm(p.mean(0) - q.mean(0))),
        }
        if loo:
            row["loo_acc"] = loo_accuracy(p, q, fit=fit, **kw)
        rows.append(row)
    return rows


def label_layers(rows: list[dict], layers: list[int] | None) -> list[dict]:
    """Attach the model layer each swept row actually came from.

    `sweep_layers` numbers its rows 0..n-1 over whatever was cached, which is only
    the model's layer index at stride 1. Under `Judge(layer_stride=4)` row 3 is
    model layer 12, so reporting the sweep's own index would mislocate every result
    in the writeup. The store manifest records the mapping; this applies it.
    """
    if not layers:
        return rows
    if len(layers) != len(rows):
        raise ValueError(f"{len(rows)} swept rows but {len(layers)} cached layers")
    return [r | {"model_layer": int(layers[r["layer"]])} for r in rows]


def best_layer(rows: list[dict], key: str = "loo_acc") -> dict:
    return max(rows, key=lambda r: (r.get(key, r["train_acc"]), r["train_acc"]))


def best_band(rows: list[dict], key: str = "loo_acc") -> dict | None:
    """Centre of the longest contiguous run of top-scoring layers.

    `best_layer` takes an argmax, and when several layers tie at the maximum the
    winner is whichever came first -- which can be an isolated spike sitting next to
    much worse layers. A layer in the middle of a run of equally good layers is a
    more robust selection, and a run is also better evidence than a spike that the
    signal is real rather than selection noise over ~60 layers.
    """
    if not rows:
        return None
    top = max(r.get(key, r["train_acc"]) for r in rows)
    best_run, run = [], []
    for r in rows:
        if r.get(key, r["train_acc"]) >= top:
            run.append(r)
            if len(run) > len(best_run):
                best_run = list(run)
        else:
            run = []
    return best_run[len(best_run) // 2] if best_run else None


def cosine(a: ProbeResult, b: ProbeResult) -> float:
    """Cosine between two probe directions -- the Phase 3.3 truth-vs-verdict control."""
    return float(_unit(a.direction) @ _unit(b.direction))
