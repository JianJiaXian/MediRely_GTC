"""Fusion modules that combine an image embedding with a (possibly pseudo)
text embedding before classification.

Provided fusion types:
  * ``concat`` : concatenate then MLP.
  * ``gated``  : learn a sigmoid gate per text dimension (handles missing text).
  * ``film``   : FiLM-style feature-wise affine conditioning of the image
                 features by the text embedding.

Every fusion module accepts ``text_present`` (B,) booleans so the model can be
trained jointly on full and missing-text samples (missing-token fusion).
"""
import torch
import torch.nn as nn


class ConcatFusion(nn.Module):
    def __init__(self, img_dim, txt_dim, hidden_dim=256, dropout=0.2):
        super().__init__()
        self.out_dim = hidden_dim
        self.mlp = nn.Sequential(
            nn.Linear(img_dim + txt_dim, hidden_dim), nn.ReLU(inplace=True),
            nn.Dropout(dropout))
        # learned embedding used when text is missing (missing-token).
        self.missing_token = nn.Parameter(torch.zeros(txt_dim))

    def forward(self, img, txt, text_present=None):
        txt = self._apply_missing(txt, text_present)
        return self.mlp(torch.cat([img, txt], dim=-1))

    def _apply_missing(self, txt, text_present):
        if text_present is None:
            return txt
        present = text_present.float().unsqueeze(-1)
        return present * txt + (1 - present) * self.missing_token


class GatedFusion(nn.Module):
    def __init__(self, img_dim, txt_dim, hidden_dim=256, dropout=0.2):
        super().__init__()
        self.out_dim = hidden_dim
        self.txt_proj = nn.Linear(txt_dim, img_dim)
        self.gate = nn.Sequential(nn.Linear(img_dim + img_dim, img_dim), nn.Sigmoid())
        self.mlp = nn.Sequential(
            nn.Linear(img_dim, hidden_dim), nn.ReLU(inplace=True), nn.Dropout(dropout))
        self.missing_token = nn.Parameter(torch.zeros(txt_dim))

    def forward(self, img, txt, text_present=None):
        if text_present is not None:
            present = text_present.float().unsqueeze(-1)
            txt = present * txt + (1 - present) * self.missing_token
        t = self.txt_proj(txt)
        g = self.gate(torch.cat([img, t], dim=-1))
        fused = img + g * t
        return self.mlp(fused)


class FiLMFusion(nn.Module):
    def __init__(self, img_dim, txt_dim, hidden_dim=256, dropout=0.2):
        super().__init__()
        self.out_dim = hidden_dim
        self.gamma = nn.Linear(txt_dim, img_dim)
        self.beta = nn.Linear(txt_dim, img_dim)
        self.mlp = nn.Sequential(
            nn.Linear(img_dim, hidden_dim), nn.ReLU(inplace=True), nn.Dropout(dropout))
        self.missing_token = nn.Parameter(torch.zeros(txt_dim))

    def forward(self, img, txt, text_present=None):
        if text_present is not None:
            present = text_present.float().unsqueeze(-1)
            txt = present * txt + (1 - present) * self.missing_token
        gamma = self.gamma(txt)
        beta = self.beta(txt)
        cond = (1 + gamma) * img + beta
        return self.mlp(cond)


def build_fusion(fusion_type, img_dim, txt_dim, hidden_dim=256, dropout=0.2):
    fusion_type = (fusion_type or "concat").lower()
    if fusion_type == "gated":
        return GatedFusion(img_dim, txt_dim, hidden_dim, dropout)
    if fusion_type == "film":
        return FiLMFusion(img_dim, txt_dim, hidden_dim, dropout)
    return ConcatFusion(img_dim, txt_dim, hidden_dim, dropout)
