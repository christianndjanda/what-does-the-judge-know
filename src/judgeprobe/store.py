"""Incremental on-disk activation storage (design doc Phase 1).

The doc's requirement is "serialise activations to disk. Do not hold a corpus in
memory." Phase 0's gates violate that -- they accumulate every activation in a list
and stack at the end, which is fine for ten transcripts and not for Phase 3.

Sizing the real thing: one activation is n_layers x d_model x 4 bytes = 1.3 MB at
64 x 5120. A debate in Phase 5 needs 3 truncations x 2 presentation orders x 2
classes = 12 of them, so ~16 MB per debate, or ~9 GB across 600 debates. That fits
on disk comfortably and in RAM only awkwardly, and it is lost entirely if the run
dies at debate 500.

So: one `.npy` per array plus a JSON manifest. Unglamorous, but it streams, it is
resumable (`has()` lets a rerun skip finished work), and a corrupt or partial file
costs one debate rather than the corpus. Reading back is `load_stacked`, which
returns exactly the `(n, n_layers, d_model)` arrays the probes expect.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


class ActivationStore:
    """Append-only activation cache backed by a directory of .npy files."""

    def __init__(self, root: Path | str, *, layers: list[int] | None = None,
                 meta: dict | None = None):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.root / "manifest.json"
        if self.manifest_path.exists():
            self.manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        else:
            self.manifest = {"layers": layers, "meta": meta or {}, "items": {}}
            self._flush()

    # --- writing --------------------------------------------------------------

    def has(self, key: str) -> bool:
        """True if this key is already written -- lets a rerun resume."""
        return key in self.manifest["items"]

    def write(self, key: str, arrays: dict[str, np.ndarray], *,
              info: dict | None = None) -> None:
        """Persist one item's arrays. `arrays` maps a name to (n_layers, d_model)."""
        safe = key.replace("/", "_")
        entry = {"files": {}, "info": info or {}}
        for name, arr in arrays.items():
            arr = np.asarray(arr, dtype=np.float32)
            path = self.root / f"{safe}__{name}.npy"
            np.save(path, arr)
            entry["files"][name] = path.name
            entry.setdefault("shape", list(arr.shape))
        self.manifest["items"][key] = entry
        self._flush()

    def _flush(self) -> None:
        self.manifest_path.write_text(json.dumps(self.manifest, indent=2),
                                      encoding="utf-8")

    # --- reading --------------------------------------------------------------

    @property
    def keys(self) -> list[str]:
        return list(self.manifest["items"])

    def load(self, key: str, name: str) -> np.ndarray:
        return np.load(self.root / self.manifest["items"][key]["files"][name])

    def load_stacked(self, name: str, keys: list[str] | None = None) -> np.ndarray:
        """`(n, n_layers, d_model)` for one array name, in `keys` order."""
        keys = keys if keys is not None else self.keys
        if not keys:
            raise ValueError(f"no items in {self.root}")
        return np.stack([self.load(k, name) for k in keys])

    def info_table(self, keys: list[str] | None = None) -> list[dict]:
        """The per-item `info` dicts, in the same order as `load_stacked`."""
        keys = keys if keys is not None else self.keys
        return [self.manifest["items"][k]["info"] | {"key": k} for k in keys]

    def __len__(self) -> int:
        return len(self.manifest["items"])

    def __repr__(self) -> str:
        return f"ActivationStore({self.root}, {len(self)} items)"
