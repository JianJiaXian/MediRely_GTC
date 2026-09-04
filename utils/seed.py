"""Reproducibility helpers."""
import os
import random

import numpy as np
import torch


def set_seed(seed: int = 42, deterministic: bool = True) -> None:
    """Seed python, numpy and torch RNGs for reproducible experiments."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        # cudnn determinism. We keep benchmark off so results are reproducible
        # at a small speed cost (acceptable for these small models).
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int) -> None:
    """Worker init fn for DataLoader to keep augmentation reproducible."""
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
