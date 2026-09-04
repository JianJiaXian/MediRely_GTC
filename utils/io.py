"""I/O helpers: YAML config loading with dotted access, CSV writing, dirs."""
import csv
import os
from typing import Any, Dict, List

import yaml


class Config(dict):
    """dict subclass allowing attribute and nested dotted access.

    Example: cfg.train.lr  or  cfg["train"]["lr"].
    """

    def __getattr__(self, key):
        try:
            value = self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc
        if isinstance(value, dict) and not isinstance(value, Config):
            value = Config(value)
            self[key] = value
        return value

    def __setattr__(self, key, value):
        self[key] = value

    def get_path(self, dotted: str, default=None):
        node: Any = self
        for part in dotted.split("."):
            if isinstance(node, dict) and part in node:
                node = node[part]
            else:
                return default
        return node


def load_config(path: str) -> Config:
    """Load a YAML config file into a Config (nested-dotted-access dict)."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path) as f:
        data = yaml.safe_load(f)
    return Config(data)


def ensure_dirs(cfg: Config) -> None:
    """Create all output directories declared in cfg.paths."""
    paths = cfg.get("paths", {})
    for key, p in paths.items():
        if isinstance(p, str):
            os.makedirs(p, exist_ok=True)


def write_csv(path: str, rows: List[Dict[str, Any]], fieldnames: List[str] = None) -> None:
    """Write a list of dicts to a CSV file (creates parent dir)."""
    if not rows:
        # still write an empty file with header if provided
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        if fieldnames:
            with open(path, "w", newline="") as f:
                csv.DictWriter(f, fieldnames=fieldnames).writeheader()
        return
    fieldnames = fieldnames or list(rows[0].keys())
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in fieldnames})


def read_csv(path: str) -> List[Dict[str, str]]:
    with open(path) as f:
        return list(csv.DictReader(f))
