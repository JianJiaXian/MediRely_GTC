"""Top-level classifier models used for training and evaluation.

Two concrete models:

  * ``ImageOnlyClassifier``  : image encoder -> MLP head.
  * ``ImageTextClassifier``  : image encoder + text encoder -> fusion -> head.

The ImageTextClassifier is the *full* model and is also the backbone reused at
inference time by the pseudo-modality bridge: we feed it a pseudo-report
embedding in place of the real text embedding. To support that, the model
exposes hooks to (a) encode an image, (b) encode text, and (c) classify from a
pair of embeddings.
"""
from typing import List

import torch
import torch.nn as nn

from models.image_encoder import build_image_encoder
from models.text_encoder import build_text_encoder
from models.fusion_models import build_fusion


class ImageOnlyClassifier(nn.Module):
    def __init__(self, cfg_model, num_classes: int):
        super().__init__()
        self.image_encoder = build_image_encoder(cfg_model)
        d = self.image_encoder.embed_dim
        hidden = int(cfg_model.fusion.get("hidden_dim", 256))
        dropout = float(cfg_model.fusion.get("dropout", 0.2))
        self.head = nn.Sequential(
            nn.Linear(d, hidden), nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.Linear(hidden, num_classes))
        self.num_classes = num_classes
        self.kind = "image_only"

    def encode_image(self, images):
        return self.image_encoder(images)

    def forward(self, batch):
        img = self.encode_image(batch["image"])
        return self.head(img)


class ImageTextClassifier(nn.Module):
    """Full image+text classifier; also the substrate for pseudo-modality."""

    def __init__(self, cfg_model, num_classes: int):
        super().__init__()
        self.image_encoder = build_image_encoder(cfg_model)
        self.text_encoder = build_text_encoder(cfg_model)
        img_dim = self.image_encoder.embed_dim
        txt_dim = self.text_encoder.embed_dim
        hidden = int(cfg_model.fusion.get("hidden_dim", 256))
        dropout = float(cfg_model.fusion.get("dropout", 0.2))
        self.fusion = build_fusion(cfg_model.fusion.get("type", "concat"),
                                   img_dim, txt_dim, hidden, dropout)
        self.head = nn.Sequential(
            nn.Linear(self.fusion.out_dim, hidden), nn.ReLU(inplace=True),
            nn.Dropout(dropout), nn.Linear(hidden, num_classes))
        self.num_classes = num_classes
        self.img_dim = img_dim
        self.txt_dim = txt_dim
        self.kind = "full_image_text"

    # --- granular hooks (used by the pseudo-modality bridge) -------------
    def encode_image(self, images) -> torch.Tensor:
        return self.image_encoder(images)

    def encode_text(self, reports: List[str]) -> torch.Tensor:
        return self.text_encoder(reports)

    def classify_from_embeddings(self, img_emb, txt_emb, text_present=None):
        fused = self.fusion(img_emb, txt_emb, text_present=text_present)
        return self.head(fused)

    def forward(self, batch, text_present=None):
        img = self.encode_image(batch["image"])
        txt = self.encode_text(batch["report"])
        if text_present is None and "has_report" in batch:
            text_present = batch["has_report"].to(img.device)
        return self.classify_from_embeddings(img, txt, text_present=text_present)

    def fit_text_vocab(self, texts):
        if hasattr(self.text_encoder, "fit_vocab"):
            self.text_encoder.fit_vocab(texts)


def build_model(cfg, num_classes: int, model_type: str = "full_image_text"):
    """Factory: ``image_only`` or ``full_image_text``."""
    if model_type == "image_only":
        return ImageOnlyClassifier(cfg.model, num_classes)
    if model_type in ("full_image_text", "full", "image_text"):
        return ImageTextClassifier(cfg.model, num_classes)
    raise ValueError(f"Unknown model_type: {model_type}")
