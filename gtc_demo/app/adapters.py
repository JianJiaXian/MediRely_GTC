"""Presentation adapter: turn a raw pipeline result into presentation-safe GUI
data. Keeps scientific inference separate from display. No filesystem paths,
patient IDs, or raw row IDs leak into the returned structure.
"""
from __future__ import annotations

import base64
import json
import math
import os

import numpy as np

from gtc_demo.app.tokens import (FINDING_NAMES, reliability_level, backend_label,
                                  RELIABILITY_TEXT, prediction_callout,
                                  CALIBRATION_LABEL)


def _finding_tag(labels, findings):
    """Short tag for an evidence card: first positive finding, else No Finding."""
    for i, f in enumerate(findings):
        if f != "no_finding" and i < len(labels) and labels[i] == 1:
            return FINDING_NAMES[f], False
    return "No Finding", True


class LogisticGate:
    """GTC MedSigLIP validation-calibrated reliability gate, reconstructed from
    saved coefficients (no sklearn at runtime). alpha = sigmoid(w·x + b) with
    x = [sim_max, sim_margin, agree]."""

    def __init__(self, coef: dict, intercept: float):
        self.w = [coef["sim_max"], coef["sim_margin"], coef["agree"]]
        self.b = float(intercept)

    @classmethod
    def from_artifact(cls, path="gtc_demo/artifacts/step4_medsiglip_comparison.json"):
        d = json.load(open(path, encoding="utf-8"))
        g = d["reliability_gate"]["medsiglip_gtc_gate"]
        return cls(g["coef"], g["intercept"])

    def predict_proba(self, X):
        out = []
        for row in X:
            z = self.b + sum(wi * xi for wi, xi in zip(self.w, row))
            p = 1.0 / (1.0 + math.exp(-z))
            out.append([1 - p, p])
        return np.asarray(out)


def img_b64(path: str) -> str:
    with open(path, "rb") as f:
        return "data:image/png;base64," + base64.b64encode(f.read()).decode()


# neutral placeholder when a retrieved neighbour image file is not staged locally
_PLACEHOLDER = ("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' "
                "width='120' height='96'><rect width='100%25' height='100%25' "
                "fill='%2303060c'/><text x='60' y='52' fill='%235f6f88' font-size='10' "
                "text-anchor='middle' font-family='sans-serif'>X-ray</text></svg>")


def safe_img_b64(path: str, image_root=None) -> str:
    """Resolve a bank image: prefer image_root/<basename>, then the original path;
    fall back to a neutral placeholder (keeps evidence real even if no thumbnail)."""
    if image_root:
        cand = os.path.join(str(image_root), os.path.basename(str(path)))
        if os.path.isfile(cand):
            return img_b64(cand)
    if path and os.path.isfile(path):
        return img_b64(path)
    return _PLACEHOLDER


def _pct(x):
    return int(round(100 * float(x)))


def to_gui_result(pred: dict, case_label: str, latency_ms: float,
                  topk_cards: int = 3, image_root=None) -> dict:
    """pred = MediRelyGTC.predict(...) output (with gate set). image_root optionally
    resolves neighbour thumbnails from a locally-staged image folder (Windows)."""
    findings = pred["findings"]
    p_img = pred["image_only_probs"]
    p_med = pred["medirely_probs"]

    # headline finding = strongest MediRely prediction. Presentation-only rule:
    # if nothing is confidently present (multi-label sigmoid all low), headline as
    # "No Finding" rather than a ~0% pathology. Probabilities are NOT modified;
    # the full findings list below still shows every real value.
    hi = max(range(len(findings)), key=lambda i: p_med[i])
    if p_med[hi] < 0.10 and findings[hi] != "no_finding" and "no_finding" in findings:
        hi = findings.index("no_finding")
    headline = FINDING_NAMES[findings[hi]]

    # full compact table (sorted by MediRely prob desc)
    order = sorted(range(len(findings)), key=lambda i: -p_med[i])
    full = [{"name": FINDING_NAMES[findings[i]], "image_only": _pct(p_img[i]),
             "medirely": _pct(p_med[i])} for i in order]

    # evidence cards (top-K real neighbours; NO ids/paths/patient info exposed)
    cards = []
    for n in pred["neighbors"][:topk_cards]:
        tag, tag_normal = _finding_tag(n.get("labels", []), findings)
        cards.append({
            "rank": n["rank"],
            "image_b64": safe_img_b64(n["image_path"], image_root),
            "similarity": round(100 * float(n["similarity"]), 1),  # one decimal
            "excerpt": _clean_excerpt(n["report_excerpt"]),
            "tag": tag, "tag_normal": tag_normal,
        })

    # recovered context (deterministic consensus over retrieved neighbours)
    neigh = pred["neighbors"]
    C = len(findings)
    counts = [0] * C
    for n in neigh:
        for c in range(C):
            counts[c] += int(n["labels"][c])
    consensus = [(FINDING_NAMES[findings[c]], counts[c]) for c in range(C)
                 if counts[c] >= max(1, (len(neigh) + 1) // 2)]
    consensus.sort(key=lambda t: -t[1])

    alpha = float(pred.get("reliability_alpha") or 0.0)
    level, level_color = reliability_level(alpha)
    blabel, is_cuvs = backend_label(pred["retrieval_backend"])
    io_pct, md_pct = _pct(p_img[hi]), _pct(p_med[hi])
    delta = md_pct - io_pct

    return {
        "case_label": case_label,
        "query_image_b64": pred.get("_query_b64"),
        "input_status": {"xray": "AVAILABLE", "report": "MISSING"},
        "evidence_cards": cards,
        "recovered_context": {
            "consensus": consensus,
            "agreeing": f"{len(neigh)}/{len(neigh)}" if consensus else f"0/{len(neigh)}",
            "n_neighbors": len(neigh),
        },
        "headline_finding": headline,
        "image_only": {"finding": headline, "pct": io_pct},
        "medirely": {"finding": headline, "pct": md_pct},
        "prediction_delta": delta,
        "prediction_callout": prediction_callout(delta),
        "full_findings": full,
        "neighbor_agreement_pct": _pct(pred["neighbor_agreement"]),
        "retrieval_confidence_pct": _pct(pred["retrieval_confidence_sim_max"]),
        "evidence_reliability_pct": _pct(alpha),
        "reliability_level": level,
        "reliability_color": level_color,
        "reliability_text": RELIABILITY_TEXT[level],
        "actual_backend": pred["retrieval_backend"],
        "backend_label": blabel,
        "is_cuvs": is_cuvs,
        "calibration_label": CALIBRATION_LABEL,
        "total_latency_ms": round(float(latency_ms), 1),
    }


def _clean_excerpt(text: str, n: int = 160) -> str:
    """Trim a REAL report excerpt for display. Only whitespace/anonymization-token
    normalisation — no rewriting/summarisation, no generation."""
    t = " ".join((text or "").split())
    t = t.replace("XXXX", "—")           # IU anonymisation placeholder → em dash
    t = t.replace("— —", "—").strip()
    if len(t) > n:
        t = t[:n].rsplit(" ", 1)[0] + "…"
    return t or "(report unavailable)"
