from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["_config_path"] = str(path)
    return cfg


def resolve_paths(cfg: dict[str, Any]) -> dict[str, Path]:
    base = Path(cfg["paths"]["data_dir"])
    artifacts = Path(cfg["paths"]["artifacts_dir"])
    paths = {"data_dir": base, "artifacts_dir": artifacts}
    for key, value in cfg["paths"].items():
        if key in {"data_dir", "artifacts_dir"}:
            continue
        paths[key] = base / value
    artifacts.mkdir(parents=True, exist_ok=True)
    return paths
