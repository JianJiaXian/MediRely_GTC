"""Classification metrics for multi-label CXR tasks.

All functions accept numpy arrays of shape (N, C):
  - ``probs``: predicted probabilities in [0, 1]
  - ``targets``: binary ground-truth labels {0, 1}

Metrics gracefully skip degenerate classes (single-class columns) for AUC.
"""
from typing import Dict

import numpy as np

try:
    from sklearn.metrics import (
        roc_auc_score,
        f1_score,
        precision_score,
        recall_score,
    )
    _HAS_SKLEARN = True
except Exception:  # pragma: no cover
    _HAS_SKLEARN = False


def _binarize(probs: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    return (probs >= threshold).astype(np.int32)


def per_class_auc(probs: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """Return per-class AUC, NaN for classes with a single ground-truth value."""
    n_classes = probs.shape[1]
    aucs = np.full(n_classes, np.nan, dtype=np.float64)
    if not _HAS_SKLEARN:
        return aucs
    for c in range(n_classes):
        y = targets[:, c]
        if y.min() == y.max():  # only one class present -> AUC undefined
            continue
        try:
            aucs[c] = roc_auc_score(y, probs[:, c])
        except ValueError:
            continue
    return aucs


def tune_thresholds(val_probs: np.ndarray, val_targets: np.ndarray,
                    grid: int = 50) -> np.ndarray:
    """Per-class decision thresholds that maximise F1 on the validation set.

    Returns an array of length C. Classes with a single value in val default to
    0.5. Used to report a fair, tuned macro F1 (AUC is threshold-independent).
    """
    val_probs = np.asarray(val_probs, dtype=np.float64)
    val_targets = np.asarray(val_targets, dtype=np.int32)
    C = val_probs.shape[1]
    thr = np.full(C, 0.5, dtype=np.float64)
    candidates = np.linspace(0.05, 0.95, grid)
    for c in range(C):
        y = val_targets[:, c]
        if y.min() == y.max():
            continue
        best_f1, best_t = -1.0, 0.5
        for t in candidates:
            pred = (val_probs[:, c] >= t).astype(np.int32)
            tp = int((pred & y).sum()); fp = int((pred & (1 - y)).sum())
            fn = int(((1 - pred) & y).sum())
            f1 = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
            if f1 > best_f1:
                best_f1, best_t = f1, t
        thr[c] = best_t
    return thr


def compute_metrics(probs: np.ndarray, targets: np.ndarray,
                    threshold=0.5) -> Dict[str, float]:
    """Compute a dictionary of multi-label metrics.

    ``threshold`` may be a scalar or a per-class array (e.g. tuned on validation).
    """
    probs = np.asarray(probs, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.int32)
    out: Dict[str, float] = {}

    aucs = per_class_auc(probs, targets)
    valid = ~np.isnan(aucs)
    out["macro_auc"] = float(np.mean(aucs[valid])) if valid.any() else float("nan")

    # micro AUC over the flattened multilabel matrix
    out["micro_auc"] = float("nan")
    if _HAS_SKLEARN and targets.min() != targets.max():
        try:
            out["micro_auc"] = float(roc_auc_score(targets.ravel(), probs.ravel()))
        except ValueError:
            pass

    thr = np.asarray(threshold, dtype=np.float64)
    preds = (probs >= thr).astype(np.int32) if thr.ndim else _binarize(probs, float(threshold))
    if _HAS_SKLEARN:
        out["macro_f1"] = float(f1_score(targets, preds, average="macro", zero_division=0))
        out["micro_f1"] = float(f1_score(targets, preds, average="micro", zero_division=0))
        out["macro_precision"] = float(
            precision_score(targets, preds, average="macro", zero_division=0))
        out["macro_recall"] = float(
            recall_score(targets, preds, average="macro", zero_division=0))
    else:  # pragma: no cover - fallback without sklearn
        tp = (preds & targets).sum()
        fp = (preds & (1 - targets)).sum()
        fn = ((1 - preds) & targets).sum()
        prec = tp / (tp + fp + 1e-8)
        rec = tp / (tp + fn + 1e-8)
        out["micro_f1"] = float(2 * prec * rec / (prec + rec + 1e-8))
        out["macro_f1"] = out["micro_f1"]
        out["macro_precision"] = float(prec)
        out["macro_recall"] = float(rec)

    out["ece"] = expected_calibration_error(probs, targets)
    return out


def expected_calibration_error(probs: np.ndarray, targets: np.ndarray,
                               n_bins: int = 15) -> float:
    """Multi-label ECE: flatten all (sample, class) probabilities and bin them.

    Returns the gap between confidence and accuracy averaged over bins,
    weighted by bin occupancy. This is an approximation suitable for reporting.
    """
    p = np.asarray(probs, dtype=np.float64).ravel()
    y = np.asarray(targets, dtype=np.int32).ravel()
    if p.size == 0:
        return float("nan")
    # confidence of the predicted (most-likely) outcome for each binary decision
    conf = np.where(p >= 0.5, p, 1.0 - p)
    correct = (np.where(p >= 0.5, 1, 0) == y).astype(np.float64)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = p.size
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (conf > lo) & (conf <= hi) if i > 0 else (conf >= lo) & (conf <= hi)
        if mask.sum() == 0:
            continue
        bin_conf = conf[mask].mean()
        bin_acc = correct[mask].mean()
        ece += (mask.sum() / n) * abs(bin_conf - bin_acc)
    return float(ece)
