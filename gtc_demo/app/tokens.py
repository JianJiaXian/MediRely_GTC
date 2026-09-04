"""Shared design tokens + presentation logic for the MediRely GTC UI.

Referenced by BOTH the Gradio CSS/HTML layer and the Pillow preview renderer so
the live app and the screenshot stay visually consistent.
"""

# ---- palette (premium dark navy radiology workstation) ------------------
COL = {
    "bg":        "#050b14",   # deep navy-black background
    "panel":     "#0b1420",   # panels
    "card":      "#111d2e",   # elevated cards
    "viewport":  "#03060c",   # near-black image viewport
    "border":    "#1e2b40",   # subtle blue-gray border
    "border2":   "#2c3e5c",   # readable edge
    "chip":      "#16233a",
    # text hierarchy (contrast-checked, recording-safe)
    "text":      "#f1f6fc",   # PRIMARY near-white
    "dim":       "#bcc8da",   # SECONDARY readable
    "muted":     "#8a99b0",   # MUTED small labels (still readable)
    "faint":     "#5f6f88",   # DISABLED only
    # accents
    "cyan":      "#38bdf8",   # primary accent (electric blue)
    "cyan_dk":   "#0ea5e9",
    "green":     "#34d399",   # HIGH reliability / trusted evidence / available
    "amber":     "#fbbf24",   # MEDIUM reliability
    "coral":     "#fb7185",   # LOW / MISSING / error
    "purple":    "#a78bfa",   # MediRely prediction comparison
}

# ---- clinical class display names ---------------------------------------
FINDING_NAMES = {
    "cardiomegaly": "Cardiomegaly", "effusion": "Pleural Effusion",
    "atelectasis": "Atelectasis", "pneumonia": "Pneumonia", "edema": "Edema",
    "pneumothorax": "Pneumothorax", "consolidation": "Consolidation",
    "no_finding": "No Finding",
}

# ---- reliability level + interpretation (GTC MedSigLIP val-calibrated alpha)
def reliability_level(alpha: float):
    if alpha >= 0.72:
        return "HIGH", "green"
    if alpha >= 0.55:
        return "MEDIUM", "amber"
    return "LOW", "coral"


RELIABILITY_TEXT = {
    "HIGH":   "Retrieved evidence is highly consistent and confident.",
    "MEDIUM": "Retrieved evidence contains some disagreement.",
    "LOW":    "Retrieved evidence is inconsistent; interpret recovered context cautiously.",
}


def prediction_callout(delta_pct: int):
    """Honest, result-dependent wording for the prediction comparison."""
    if delta_pct >= 10:
        return "MediRely strengthens the top prediction using recovered evidence."
    if delta_pct <= -10:
        return "Recovered context lowers the top-prediction confidence."
    return "Prediction updated after recovered context."


def backend_label(backend: str):
    """Honest backend label. NEVER claim NVIDIA cuVS unless backend == 'cuvs'."""
    if backend == "cuvs":
        return "NVIDIA cuVS", True
    if backend == "raft":
        return "NVIDIA RAFT (cuVS fallback)", False
    if backend == "torch":
        return "PyTorch CUDA", False        # native-Windows local-GPU fallback
    if backend in ("faiss-gpu", "faiss-cpu"):
        return backend.upper(), False
    return f"{backend} (fallback)", False


# ---- small, secondary scientific credibility strip (REAL Step-4 values) --
# Source: gtc_demo/artifacts/step4_medsiglip_comparison.json (held-out IU X-Ray
# test, n=1517). Labeled as the GTC extension, NOT the original MICCAI result.
MODEL_AUC = {"image_only": 0.722, "medirely": 0.819,
             "source": "Held-out IU X-Ray test · macro AUC"}

CALIBRATION_LABEL = "GTC extension • calibrated on validation split"
RESEARCH_REFERENCE = ("Original MediRely uses the validated XRV-based reliability "
                      "path. This demo's evidence reliability is the GTC MedSigLIP "
                      "retrieval extension, calibrated on the validation split.")
DISCLAIMER = "Research prototype • Not for clinical use"
MICCAI_NOTE = "Research accepted at MICCAI 2026 ML-CDS Workshop"
