"""Load configs/default.yaml. Import CFG everywhere instead of hardcoding constants.  Owner: P1"""
from __future__ import annotations

import os
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent


def load_config(path: str | Path | None = None) -> dict:
    path = Path(path) if path else ROOT / "configs" / "default.yaml"
    with open(path) as f:
        cfg = yaml.safe_load(f)
    # Demo day: point the API at the evaluator trips without editing code.
    if os.environ.get("PATHSNAP_EVAL_DIR"):
        cfg["paths"]["eval_dir"] = os.environ["PATHSNAP_EVAL_DIR"]
    # Resolve relative paths against repo root.
    for k, v in cfg["paths"].items():
        p = Path(v)
        cfg["paths"][k] = str(p if p.is_absolute() else ROOT / p)
    return cfg


CFG = load_config()
