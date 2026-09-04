"""Lightweight logging utilities: console + file logger and a JSONL metric log."""
import json
import logging
import os
import sys
from datetime import datetime


def get_logger(name: str = "mlcds", log_dir: str = None, exp_name: str = None) -> logging.Logger:
    """Return a logger that writes both to stdout and (optionally) a file."""
    logger = logging.getLogger(name)
    if logger.handlers:  # already configured
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("[%(asctime)s] %(levelname)s %(message)s", "%H:%M:%S")

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    if log_dir is not None:
        os.makedirs(log_dir, exist_ok=True)
        tag = exp_name or name
        fpath = os.path.join(log_dir, f"{tag}.log")
        fh = logging.FileHandler(fpath)
        fh.setFormatter(fmt)
        logger.addHandler(fh)
        logger.info("Logging to %s", fpath)
    return logger


class MetricLogger:
    """Appends per-epoch metric dicts to a JSONL file for later plotting."""

    def __init__(self, path: str):
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)

    def log(self, record: dict) -> None:
        record = dict(record)
        record.setdefault("timestamp", datetime.now().isoformat(timespec="seconds"))
        with open(self.path, "a") as f:
            f.write(json.dumps(record) + "\n")
