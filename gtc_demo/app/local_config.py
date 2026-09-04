"""Local-GPU (Windows) configuration layer.

Resolves artifact / model / image locations via environment variables with
pathlib, defaulting to a self-contained ``artifacts_local`` bundle at the project
root. Cluster / H200 paths are NOT touched — this layer is only used by
``--local-gpu`` mode.

Environment overrides:
  MEDIRELY_ARTIFACT_ROOT   folder holding the exported validated artifacts
                           (default: <project>/artifacts_local)
  MEDIRELY_MODEL_ROOT      folder holding the MedSigLIP snapshot
                           (default: <project>/models)
  MEDIRELY_IMAGE_ROOT      optional folder of training X-rays for neighbour
                           thumbnails (default: <artifact_root>/bank_images)
"""
from __future__ import annotations

import os
from pathlib import Path

_PROJECT = Path(__file__).resolve().parents[2]     # .../miccai_mlcds


def _p(env, default: Path) -> Path:
    v = os.environ.get(env)
    return Path(v) if v else default


class LocalConfig:
    def __init__(self):
        self.artifact_root = _p("MEDIRELY_ARTIFACT_ROOT", _PROJECT / "artifacts_local")
        # default MedSigLIP at <project>/medsiglip-448 — a SIBLING of the validated
        # `models/` python package (never inside it, to avoid import collisions).
        self.model_root = _p("MEDIRELY_MODEL_ROOT", _PROJECT)
        self.image_root = _p("MEDIRELY_IMAGE_ROOT", self.artifact_root / "bank_images")

    # ---- required artifacts (fall back to repo defaults if bundle absent) ----
    def _art(self, name, repo_default):
        c = self.artifact_root / name
        return c if c.exists() else (_PROJECT / repo_default)

    def config_yaml(self):
        return str(self._art("iu_xray_xrv.yaml", "configs/iu_xray_xrv.yaml"))

    def checkpoint(self):
        return str(self._art("best_iu_full_image_text_xrv.pth",
                             "outputs/checkpoints/best_iu_full_image_text_xrv.pth"))

    def xrv_bank(self):
        return str(self._art("iu_train_memory_bank_xrv.pt",
                             "outputs/memory_banks/iu_train_memory_bank_xrv.pt"))

    def ms_emb(self):
        return str(self._art("medsiglip_train_embeddings.npz",
                             "gtc_demo/artifacts/medsiglip_train_embeddings.npz"))

    def comparison(self):
        return str(self._art("step4_medsiglip_comparison.json",
                             "gtc_demo/artifacts/step4_medsiglip_comparison.json"))

    def gallery_sources(self):
        return str(self._art("gallery_sources.json",
                             "gtc_demo/artifacts/gallery_sources.json"))

    def ms_model_path(self):
        return str(self.model_root / "medsiglip-448")

    def image_root_or_none(self):
        return str(self.image_root) if self.image_root.exists() else None

    # ---- startup validation ----
    def missing_required(self):
        req = {"config": self.config_yaml(), "checkpoint": self.checkpoint(),
               "xrv_bank": self.xrv_bank(), "medsiglip_embeddings": self.ms_emb(),
               "reliability_calibration": self.comparison(),
               "gallery_sources": self.gallery_sources(),
               "medsiglip_model_dir": self.ms_model_path()}
        return {k: v for k, v in req.items() if not Path(v).exists()}

    def summary(self):
        return {"artifact_root": str(self.artifact_root), "model_root": str(self.model_root),
                "image_root": self.image_root_or_none()}
