"""The pseudo-modality bridge: the paper's main method.

This module is *training-free*. It wraps an already-trained
``ImageTextClassifier`` and a memory bank built from the training set. At
inference, for a test image whose report is missing, it:

  1. encodes the test image -> image embedding,
  2. retrieves the top-k most similar training images from the memory bank,
  3. aggregates their *report* embeddings into a pseudo-report embedding
     (similarity-weighted by default),
  4. optionally falls back to a safe default when retrieval confidence is low,
  5. fuses the image embedding with the pseudo-report embedding using the
     trained fusion head to produce a prediction.

It also implements the comparison baselines (zero / mean / random text,
top-1/3/5) by selecting the retrieval ``mode``.
"""
from dataclasses import dataclass
from typing import Dict, List

import torch

from utils.retrieval import MemoryBankRetriever, GroupPrototypeRetriever


# Mapping from experiment name -> (retrieval mode, fallback override).
EXPERIMENT_MODES = {
    "image_only": ("image_only", None),          # handled specially (no text)
    "full_image_text": ("full", None),           # uses real report (upper bound)
    "zero_text": ("zero_text", "none"),
    "mean_report": ("mean_report", "none"),
    "random_report": ("random_report", "none"),
    "top1": ("top1", "none"),
    "top3": ("top3", "none"),
    "top5": ("top5", "none"),
    "similarity_weighted": ("similarity_weighted", "none"),
    "threshold_fallback": ("similarity_weighted", "threshold"),
    "mean_fallback": ("similarity_weighted", "mean"),
    "image_only_fallback": ("similarity_weighted", "image_only"),
    # Grouping-based pseudo-modality (advisor-suggested variant).
    "group_prototype": ("group_prototype", "none"),
    "group_prototype_fallback": ("group_prototype", "threshold"),
    # Retrieval label-prior variants: borrow neighbours' LABELS (the most direct
    # recovered signal) to push missing-modality accuracy toward full-modality.
    "label_prior": ("similarity_weighted", "none"),
    "pseudo_label_fusion": ("similarity_weighted", "none"),
}


@dataclass
class MemoryBank:
    image_embeddings: torch.Tensor   # (N, Di)
    text_embeddings: torch.Tensor    # (N, Dt)
    labels: torch.Tensor             # (N, C)
    report_texts: List[str]
    image_paths: List[str]
    meta: dict

    def save(self, path: str):
        torch.save({
            "image_embeddings": self.image_embeddings,
            "text_embeddings": self.text_embeddings,
            "labels": self.labels,
            "report_texts": self.report_texts,
            "image_paths": self.image_paths,
            "meta": self.meta,
        }, path)

    @staticmethod
    def load(path: str) -> "MemoryBank":
        d = torch.load(path, map_location="cpu", weights_only=False)
        return MemoryBank(
            d["image_embeddings"], d["text_embeddings"], d["labels"],
            d["report_texts"], d["image_paths"], d.get("meta", {}))


class PseudoModalityBridge:
    """Inference-time bridge wrapping a trained model + memory bank."""

    def __init__(self, model, memory_bank: MemoryBank, retrieval_cfg,
                 device: str = "cuda"):
        self.model = model.to(device).eval()
        self.device = device
        self.bank = memory_bank
        rc = retrieval_cfg
        self.retriever = MemoryBankRetriever(
            image_bank=memory_bank.image_embeddings,
            text_bank=memory_bank.text_embeddings,
            metric=rc.get("metric", "cosine"),
            weighting=rc.get("weighting", "similarity"),
            temperature=float(rc.get("softmax_temperature", 0.1)),
            normalize=bool(rc.get("normalize_embeddings", True)),
            device=device,
            seed=42,
        )
        self.default_topk = int(rc.get("topk", 5))
        self.default_threshold = float(rc.get("similarity_threshold", 0.5))
        self.softmax_temperature = float(rc.get("softmax_temperature", 0.1))
        # Neighbour labels for the retrieval label-prior path.
        self.bank_labels = memory_bank.labels.to(device).float()  # (N, C)
        self.label_fusion_weight = float(rc.get("label_fusion_weight", 0.5))
        # Lazily-built grouping retriever (advisor-suggested variant).
        self._group_retriever = None
        self._group_cfg = {
            "strategy": rc.get("group_strategy", "kmeans"),
            "n_groups": int(rc.get("n_groups", 16)),
            "normalize": bool(rc.get("normalize_embeddings", True)),
        }

    def _ensure_group_retriever(self):
        if self._group_retriever is None:
            self._group_retriever = GroupPrototypeRetriever(
                image_bank=self.bank.image_embeddings,
                text_bank=self.bank.text_embeddings,
                labels=self.bank.labels,
                strategy=self._group_cfg["strategy"],
                n_groups=self._group_cfg["n_groups"],
                normalize=self._group_cfg["normalize"],
                device=self.device, seed=42)
        return self._group_retriever

    @torch.no_grad()
    def predict_batch(self, images: torch.Tensor, reports: List[str],
                      experiment: str = "similarity_weighted",
                      topk: int = None, threshold: float = None,
                      collect_retrievals: bool = False) -> Dict:
        """Run one experiment configuration on a batch.

        Returns dict with ``logits`` (B, C) and optional retrieval metadata.
        """
        images = images.to(self.device)
        img_emb = self.model.encode_image(images)        # (B, Di)
        B = img_emb.shape[0]
        mode, fb_override = EXPERIMENT_MODES.get(
            experiment, ("similarity_weighted", "none"))
        topk = topk or self.default_topk
        threshold = self.default_threshold if threshold is None else threshold

        # --- special cases that bypass retrieval -------------------------
        if experiment == "image_only":
            # No text path: use the missing-token (text_present=False).
            zero_txt = torch.zeros(B, self.model.txt_dim, device=self.device)
            present = torch.zeros(B, dtype=torch.bool, device=self.device)
            logits = self.model.classify_from_embeddings(img_emb, zero_txt, present)
            return {"logits": logits.cpu(), "used_fallback": [False] * B,
                    "max_sim": [0.0] * B, "retrievals": []}

        if experiment == "full_image_text" or mode == "full":
            txt_emb = self.model.encode_text(reports)
            present = torch.tensor(
                [bool(r.strip()) for r in reports], device=self.device)
            logits = self.model.classify_from_embeddings(img_emb, txt_emb, present)
            return {"logits": logits.cpu(), "used_fallback": [False] * B,
                    "max_sim": [1.0] * B, "retrievals": []}

        # --- retrieval-based pseudo-modality -----------------------------
        fallback = fb_override if fb_override is not None else "none"
        pseudo = torch.zeros(B, self.model.txt_dim, device=self.device)
        used_fb, max_sims, retr_meta = [], [], []
        present = torch.ones(B, dtype=torch.bool, device=self.device)
        group_retr = self._ensure_group_retriever() if mode == "group_prototype" else None
        need_label_prior = experiment in ("label_prior", "pseudo_label_fusion")
        knn_prior = torch.zeros(B, self.bank_labels.shape[1], device=self.device)
        for i in range(B):
            if group_retr is not None:
                res = group_retr.retrieve(
                    img_emb[i], fallback=fallback, similarity_threshold=threshold)
            else:
                res = self.retriever.retrieve(
                    img_emb[i], mode=mode, topk=topk, fallback=fallback,
                    similarity_threshold=threshold)
            pseudo[i] = res.pseudo_text.to(self.device)
            used_fb.append(res.used_fallback)
            max_sims.append(res.max_sim)
            if res.used_fallback and fallback in ("image_only",):
                present[i] = False
            if need_label_prior and res.topk_indices.numel() > 0:
                idx = res.topk_indices.to(self.device)
                sims = res.topk_sims.to(self.device).float()
                w = torch.softmax(sims / max(self.softmax_temperature, 1e-6), dim=0)
                knn_prior[i] = (w.unsqueeze(1) * self.bank_labels[idx]).sum(0)
            if collect_retrievals:
                retr_meta.append({
                    "topk_indices": res.topk_indices.tolist(),
                    "topk_sims": [round(float(s), 4) for s in res.topk_sims.tolist()],
                    "max_sim": round(res.max_sim, 4),
                    "used_fallback": res.used_fallback,
                    "fallback_type": res.fallback_type,
                })

        # Logit assembly depends on the experiment.
        def _logit(p):
            p = p.clamp(1e-4, 1 - 1e-4)
            return torch.log(p / (1 - p))

        if experiment == "label_prior":
            logits = _logit(knn_prior)
        elif experiment == "pseudo_label_fusion":
            pseudo_logits = self.model.classify_from_embeddings(img_emb, pseudo, present)
            wgt = self.label_fusion_weight
            p = (1 - wgt) * torch.sigmoid(pseudo_logits) + wgt * knn_prior
            logits = _logit(p)
        else:
            logits = self.model.classify_from_embeddings(img_emb, pseudo, present)
        return {"logits": logits.cpu(), "used_fallback": used_fb,
                "max_sim": max_sims, "retrievals": retr_meta}
