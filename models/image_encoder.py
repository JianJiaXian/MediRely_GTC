"""Lightweight image encoders.

Default: torchvision ResNet18 with an optional ImageNet-pretrained init that
*gracefully* falls back to random weights if the download/weights are not
available (offline SLURM nodes). A tiny custom CNN is also provided so the code
runs with zero torchvision model-zoo dependency.

All encoders expose ``.embed_dim`` and return an L2-normalisable feature vector
of shape (B, embed_dim).
"""
import torch
import torch.nn as nn


class SmallCNN(nn.Module):
    """A minimal 4-block CNN encoder (no pretrained weights needed)."""

    def __init__(self, embed_dim: int = 256, in_ch: int = 3):
        super().__init__()
        def block(i, o):
            return nn.Sequential(
                nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
                nn.Conv2d(o, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
                nn.MaxPool2d(2))
        self.features = nn.Sequential(
            block(in_ch, 32), block(32, 64), block(64, 128), block(128, 256))
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Linear(256, embed_dim)
        self.embed_dim = embed_dim

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x).flatten(1)
        return self.proj(x)


class ResNetEncoder(nn.Module):
    """ResNet18/50 backbone with a projection head to ``embed_dim``."""

    def __init__(self, backbone: str = "resnet18", pretrained: bool = True,
                 embed_dim: int = 256):
        super().__init__()
        from torchvision import models

        weights = None
        feat_dim = 512
        try:
            if backbone == "resnet50":
                w = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
                net = models.resnet50(weights=w)
                feat_dim = 2048
            else:
                w = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
                net = models.resnet18(weights=w)
                feat_dim = 512
            weights = w
        except Exception as exc:  # offline / weights missing -> random init
            print(f"[image_encoder] pretrained weights unavailable ({exc}); "
                  f"using random init for {backbone}.")
            if backbone == "resnet50":
                net = models.resnet50(weights=None)
                feat_dim = 2048
            else:
                net = models.resnet18(weights=None)
                feat_dim = 512
        if weights is None and pretrained:
            print("[image_encoder] NOTE: running with randomly-initialised backbone.")

        net.fc = nn.Identity()
        self.backbone = net
        self.proj = nn.Linear(feat_dim, embed_dim)
        self.embed_dim = embed_dim

    def forward(self, x):
        feat = self.backbone(x)
        return self.proj(feat)


class FoundationEmbeddingEncoder(nn.Module):
    """Pluggable wrapper around a frozen medical foundation embedding model.

    Motivated by advisor feedback to consider strong medical embeddings such as
    **MedImageInsight** for the retrieval backbone: better image embeddings give
    better nearest neighbours and therefore a better pseudo-report, *without*
    changing the rest of the training-free bridge.

    The actual foundation model (e.g. MedImageInsight) is loaded lazily via a
    user-supplied callable / checkpoint. Because those weights are large and may
    be unavailable offline, this class FAILS GRACEFULLY: if the backbone cannot
    be loaded it falls back to a torchvision ResNet and prints a clear note, so
    the pipeline always runs. A small trainable projection adapts the (frozen)
    foundation features to ``embed_dim`` and is the only part updated by
    training.
    """

    def __init__(self, embed_dim: int = 256, feature_dim: int = 1024,
                 weights_path: str = None, freeze: bool = True):
        super().__init__()
        self.embed_dim = embed_dim
        self.freeze = freeze
        self.backbone = None
        self._feature_dim = feature_dim
        try:
            self.backbone = self._load_foundation(weights_path)
            print(f"[image_encoder] loaded foundation embedding model "
                  f"({weights_path}).")
        except Exception as exc:
            print(f"[image_encoder] foundation model unavailable ({exc}); "
                  f"falling back to ResNet18. To use MedImageInsight, implement "
                  f"_load_foundation() and provide weights.")
            self._fallback = ResNetEncoder("resnet18", pretrained=True,
                                           embed_dim=embed_dim)
            self.proj = nn.Identity()
            return
        if freeze:
            for p in self.backbone.parameters():
                p.requires_grad = False
        self.proj = nn.Linear(self._feature_dim, embed_dim)
        self._fallback = None

    def _load_foundation(self, weights_path):
        # Placeholder hook. To enable MedImageInsight, load the model here and
        # return an nn.Module mapping (B,3,H,W) -> (B, feature_dim). We raise by
        # default so the graceful fallback is exercised until weights are wired.
        raise NotImplementedError("MedImageInsight backbone not wired yet.")

    def forward(self, x):
        if self._fallback is not None:
            return self._fallback(x)
        feat = self.backbone(x)
        if self.freeze:
            feat = feat.detach()
        return self.proj(feat)


# ImageNet stats used by utils/transforms (needed to invert normalization
# before feeding a foundation model that expects a different input range).
_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


class TorchXRayVisionEncoder(nn.Module):
    """Frozen TorchXRayVision DenseNet-121 as a medical (CXR) foundation encoder.

    The backbone is a DenseNet-121 pretrained jointly on NIH, PadChest, CheXpert,
    MIMIC-CXR, Google, OpenI and Kaggle chest X-rays -- a genuine multi-dataset
    CXR foundation model. It outputs a 1024-d pooled feature which we adapt to
    ``embed_dim`` with a small trainable projection (the only trained part).

    Our dataloader feeds 3-channel ImageNet-normalised images; XRV instead
    expects a single channel in roughly [-1024, 1024]. We invert the ImageNet
    normalisation, convert to grayscale, and rescale internally so the
    foundation model receives correctly-formatted input.

    Falls back to a ResNet18 encoder if torchxrayvision / weights are
    unavailable, so the pipeline always runs.
    """

    def __init__(self, embed_dim: int = 256, freeze: bool = True,
                 weights: str = "densenet121-res224-all"):
        super().__init__()
        self.embed_dim = embed_dim
        self.freeze = freeze
        self._fallback = None
        try:
            import torchxrayvision as xrv
            self.xrv = xrv
            self.backbone = xrv.models.DenseNet(weights=weights)
            self._feat_dim = 1024
            print(f"[image_encoder] loaded TorchXRayVision foundation backbone "
                  f"({weights}).")
        except Exception as exc:
            print(f"[image_encoder] torchxrayvision unavailable ({exc}); "
                  f"falling back to ResNet18.")
            self._fallback = ResNetEncoder("resnet18", pretrained=True,
                                           embed_dim=embed_dim)
            self.proj = nn.Identity()
            return
        if freeze:
            self.backbone.eval()
            for p in self.backbone.parameters():
                p.requires_grad = False
        self.proj = nn.Linear(self._feat_dim, embed_dim)
        self.register_buffer("imagenet_mean", _IMAGENET_MEAN.clone())
        self.register_buffer("imagenet_std", _IMAGENET_STD.clone())

    def _to_xrv_input(self, x):
        # x: (B,3,H,W) ImageNet-normalised -> de-normalise to [0,1]
        x01 = x * self.imagenet_std + self.imagenet_mean
        x01 = x01.clamp(0, 1).mean(dim=1, keepdim=True)        # grayscale (B,1,H,W)
        # XRV expects roughly [-1024, 1024]
        return x01 * 2048.0 - 1024.0

    def forward(self, x):
        if self._fallback is not None:
            return self._fallback(x)
        xin = self._to_xrv_input(x)
        if self.freeze:
            self.backbone.eval()
            with torch.no_grad():
                feat = self.backbone.features2(xin)            # (B,1024)
            feat = feat.detach()
        else:
            feat = self.backbone.features2(xin)
        return self.proj(feat)


def build_image_encoder(cfg_model) -> nn.Module:
    ie = cfg_model.image_encoder
    backbone = ie.get("backbone", "resnet18")
    embed_dim = int(ie.get("embed_dim", 256))
    if backbone == "smallcnn":
        return SmallCNN(embed_dim=embed_dim)
    if backbone in ("torchxrayvision", "xrv"):
        return TorchXRayVisionEncoder(
            embed_dim=embed_dim,
            freeze=bool(ie.get("freeze", True)),
            weights=ie.get("weights", "densenet121-res224-all"))
    if backbone in ("medimageinsight", "foundation"):
        return FoundationEmbeddingEncoder(
            embed_dim=embed_dim,
            feature_dim=int(ie.get("feature_dim", 1024)),
            weights_path=ie.get("weights_path", None),
            freeze=bool(ie.get("freeze", True)))
    return ResNetEncoder(backbone=backbone,
                         pretrained=bool(ie.get("pretrained", True)),
                         embed_dim=embed_dim)
