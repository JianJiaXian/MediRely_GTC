"""Frozen MedSigLIP-448 image encoder for GTC retrieval (Step 3).

Uses the OFFICIAL model + preprocessing via HF ``AutoModel``/``AutoProcessor``
loaded from a locally-staged snapshot (no re-download at run time). We do NOT
reuse the validated XRV normalization here -- MedSigLIP has its own preprocessing
(SiglipImageProcessor: 448x448 bicubic resize, rescale 1/255, mean/std = 0.5).

The encoder is frozen and used only to produce image embeddings for retrieval:
  * encode_image(pil_image) -> (D,) float32
  * encode_batch(list[pil]) -> (B, D) float32
Embeddings are returned RAW (un-normalized); the retrieval layer L2-normalizes
(exactly as the Step-2 XRV bank did), so index treatment is identical across
representations. A ``normalize=True`` option is provided for convenience.

Grayscale CXRs are converted L->RGB (channel replication) before the official
processor, which is the standard way to feed a single-channel X-ray to an RGB
vision-language tower. Everything runs under ``torch.inference_mode()`` on GPU.
"""
from __future__ import annotations

import time
from typing import List

import numpy as np


class MedSigLIPEncoder:
    MODEL_ID = "google/medsiglip-448"

    def __init__(self, model_path: str, device: str = None,
                 dtype: str = "float32"):
        import torch
        from transformers import AutoModel, AutoImageProcessor
        self._torch = torch
        self.model_path = model_path
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16,
                       "float16": torch.float16}[dtype]
        self.dtype_str = dtype

        # Official IMAGE preprocessing only (SiglipImageProcessor). We use the
        # vision tower via get_image_features; the text tokenizer (which needs
        # SentencePiece) is intentionally not loaded -- no text is encoded here.
        self.processor = AutoImageProcessor.from_pretrained(model_path)
        self.model = AutoModel.from_pretrained(
            model_path, torch_dtype=self._dtype).to(self.device).eval()
        for p in self.model.parameters():
            p.requires_grad = False

        # image size / mean / std from the official processor (report, not guess)
        ip = getattr(self.processor, "image_processor", self.processor)
        self.image_size = getattr(ip, "size", None)
        self.image_mean = getattr(ip, "image_mean", None)
        self.image_std = getattr(ip, "image_std", None)
        self._embed_dim = None  # discovered on first forward

    # ---------------------------------------------------------------- #
    def _to_rgb(self, img):
        from PIL import Image
        if not isinstance(img, Image.Image):
            img = Image.open(img)
        return img.convert("RGB")  # grayscale CXR -> 3-channel replicate

    def _forward(self, pil_list: List):
        torch = self._torch
        rgb = [self._to_rgb(im) for im in pil_list]
        inputs = self.processor(images=rgb, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self.device, dtype=self._dtype)
        with torch.inference_mode():
            feats = self.model.get_image_features(pixel_values=pixel_values)
        feats = feats.float()  # always emit float32 for the index
        if self._embed_dim is None:
            self._embed_dim = int(feats.shape[-1])
        return feats, tuple(pixel_values.shape)

    @property
    def embed_dim(self):
        return self._embed_dim

    def encode_batch(self, pil_list: List, normalize: bool = False) -> np.ndarray:
        feats, _ = self._forward(pil_list)
        if normalize:
            feats = self._torch.nn.functional.normalize(feats, p=2, dim=-1, eps=1e-8)
        return feats.detach().cpu().numpy().astype(np.float32)

    def encode_image(self, pil_image, normalize: bool = False) -> np.ndarray:
        return self.encode_batch([pil_image], normalize=normalize)[0]

    # diagnostics ----------------------------------------------------- #
    def probe(self, pil_image):
        """Return a dict of shapes/dtype/norm/nan-inf/latency for the smoke test."""
        torch = self._torch
        if self.device == "cuda":
            torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        feats, px_shape = self._forward([self._to_rgb(pil_image)])
        if self.device == "cuda":
            torch.cuda.synchronize()
        latency_ms = (time.perf_counter() - t0) * 1000.0
        emb = feats.detach().cpu().numpy().astype(np.float32)[0]
        gpu_mem_mb = None
        if self.device == "cuda":
            gpu_mem_mb = round(torch.cuda.max_memory_allocated() / 1e6, 1)
        return {
            "processed_tensor_shape": list(px_shape),
            "embedding_shape": list(emb.shape),
            "embedding_dtype": str(emb.dtype),
            "l2_norm": float(np.linalg.norm(emb)),
            "has_nan": bool(np.isnan(emb).any()),
            "has_inf": bool(np.isinf(emb).any()),
            "inference_latency_ms": round(latency_ms, 3),
            "gpu_peak_mem_mb": gpu_mem_mb,
            "device": self.device,
            "model_dtype": self.dtype_str,
        }
