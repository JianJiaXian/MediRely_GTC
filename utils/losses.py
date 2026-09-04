"""Loss functions for multi-label classification."""
import torch
import torch.nn as nn


def build_loss(name: str = "bce", pos_weight: torch.Tensor = None) -> nn.Module:
    """Factory for the training loss.

    ``bce``   -> BCEWithLogitsLoss (default, multi-label).
    ``focal`` -> a simple multi-label focal loss for class imbalance.
    """
    name = (name or "bce").lower()
    if name == "bce":
        return nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    if name == "focal":
        return FocalBCELoss(pos_weight=pos_weight)
    raise ValueError(f"Unknown loss: {name}")


class FocalBCELoss(nn.Module):
    """Multi-label focal loss built on top of BCE-with-logits."""

    def __init__(self, gamma: float = 2.0, pos_weight: torch.Tensor = None):
        super().__init__()
        self.gamma = gamma
        self.register_buffer(
            "pos_weight", pos_weight if pos_weight is not None else None
        ) if pos_weight is not None else setattr(self, "pos_weight", None)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = nn.functional.binary_cross_entropy_with_logits(
            logits, targets, reduction="none", pos_weight=self.pos_weight
        )
        p = torch.sigmoid(logits)
        p_t = p * targets + (1 - p) * (1 - targets)
        focal = (1 - p_t).clamp(min=1e-6) ** self.gamma
        return (focal * bce).mean()
