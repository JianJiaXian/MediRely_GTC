"""MediRely GTC inference orchestrator.

Swaps ONLY the retrieval representation; the prediction/classifier branch and the
pseudo-report aggregation + reliability *formulas* are the validated ones, reused
read-only from the MICCAI tree:
  * XRV image encoder + multimodal classifier  (models/…)
  * aggregate_pseudo_text (softmax(sim/T))      (utils/retrieval.py)
  * MemoryBankRetriever (validated retrieval)   (utils/retrieval.py)
  * neighbour-agreement / sim_max / sim_margin  (as in run_reliability_experiments)

Supported retrieval configs:
  A  xrv       / validated  -> validated MemoryBankRetriever over XRV bank
  B  xrv       / cuvs       -> genuine cuVS over XRV bank
  C  medsiglip / cuvs       -> genuine cuVS over MedSigLIP bank (shared row IDs)

The classifier ALWAYS consumes the XRV image embedding + a pseudo-report built
from the XRV-bank text embeddings. MedSigLIP only re-ranks which real neighbours
are selected.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import torch

RETRIEVAL_CONFIGS = {
    "A": {"encoder": "xrv", "backend": "validated"},
    "B": {"encoder": "xrv", "backend": "cuvs"},
    "C": {"encoder": "medsiglip", "backend": "cuvs"},
}


def _load_checkpoint(path, model, device="cpu"):
    """Identical to train.load_checkpoint — inlined so Local-GPU (Windows) does not
    need train.py / datasets/. Restores the BoW vocab then the model weights."""
    state = torch.load(path, map_location=device, weights_only=False)
    if "text_vocab" in state and hasattr(model, "text_encoder") and \
            hasattr(model.text_encoder, "import_vocab"):
        model.text_encoder.import_vocab(state["text_vocab"])
    model.load_state_dict(state["model_state"])
    return state


@dataclass
class RetrievalOut:
    idx: np.ndarray      # (B,k) row IDs into the shared bank
    sims: np.ndarray     # (B,k) cosine similarities (encoder-specific scale)
    backend: str
    encoder: str


class MediRelyGTC:
    def __init__(self, config_path="configs/iu_xray_xrv.yaml",
                 checkpoint="outputs/checkpoints/best_iu_full_image_text_xrv.pth",
                 xrv_bank="outputs/memory_banks/iu_train_memory_bank_xrv.pt",
                 ms_emb="gtc_demo/artifacts/medsiglip_train_embeddings.npz",
                 ms_model_path=os.environ.get("MEDIRELY_MEDSIGLIP_DIR", "models/medsiglip-448"),
                 topk=10, device=None, load_medsiglip=True,
                 retrieval_prefer="cuvs", ms_dtype="float32",
                 retrieval_allow_fallback=True):
        # retrieval_allow_fallback=False makes an EXPLICIT cuVS request strict:
        # if genuine cuVS cannot import/build, construction raises a clear
        # RuntimeError instead of silently using PyTorch.
        self._retrieval_allow_fallback = retrieval_allow_fallback
        # retrieval_prefer/ms_dtype default to the H200 contest behaviour; the
        # Windows Local-GPU mode passes prefer="cuvs" (auto-falls back to torch when
        # cuVS is unavailable, e.g. native Windows) and reports the ACTUAL backend.
        self._retrieval_prefer = retrieval_prefer
        self._ms_dtype = ms_dtype
        from utils.io import load_config
        from models.classifiers import build_model
        from models.pseudo_modality_bridge import MemoryBank
        from utils.retrieval import MemoryBankRetriever, aggregate_pseudo_text
        from gtc_demo.retrieval import build_retriever
        # NB: checkpoint loading is inlined below (see _load_checkpoint) — identical
        # to train.load_checkpoint — so Windows Local-GPU needs neither train.py nor
        # datasets/. H200 behaviour is unchanged.

        self._agg = aggregate_pseudo_text
        self.cfg = load_config(config_path)
        self.topk = int(self.cfg.retrieval.get("topk", topk))
        self.temperature = float(self.cfg.retrieval.get("softmax_temperature", 0.1))
        self.weighting = self.cfg.retrieval.get("weighting", "similarity")
        self.sim_threshold = float(self.cfg.retrieval.get("similarity_threshold", 0.5))
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # validated model + bank
        self.model = build_model(self.cfg, len(self.cfg.dataset.findings),
                                 "full_image_text").to(self.device).eval()
        _load_checkpoint(checkpoint, self.model, self.device)
        self.bank = MemoryBank.load(xrv_bank)
        self.text_bank = self.bank.text_embeddings.to(self.device).float()   # (N,256)
        self.labels = self.bank.labels.float()                               # (N,8) cpu
        self.paths = [str(p) for p in self.bank.image_paths]
        self.reports = self.bank.report_texts
        self.findings = list(self.cfg.dataset.findings)

        # validated retriever (config A)
        self.membank = MemoryBankRetriever(
            image_bank=self.bank.image_embeddings, text_bank=self.bank.text_embeddings,
            metric=self.cfg.retrieval.get("metric", "cosine"),
            weighting=self.weighting, temperature=self.temperature,
            normalize=bool(self.cfg.retrieval.get("normalize_embeddings", True)),
            device=self.device, seed=42)

        # cuVS over XRV bank (config B)
        xrv_np = self.bank.image_embeddings.detach().cpu().numpy().astype(np.float32)
        self.cuvs_xrv = build_retriever(prefer=self._retrieval_prefer, allow_fallback=self._retrieval_allow_fallback).build(xrv_np)
        self.xrv_backend = self.cuvs_xrv.backend

        # MedSigLIP bank + cuVS (config C); assert row alignment
        self.ms_ready = False
        self.ms_backend = None
        if load_medsiglip and os.path.isfile(ms_emb):
            d = np.load(ms_emb, allow_pickle=True)
            # Row alignment: the local (private) npz carries absolute image_paths, so
            # we assert exact path-order match against the XRV bank. The sanitized
            # public npz omits image_paths (privacy) but preserves row order + row_ids,
            # so we fall back to a count + row_id-checksum check. Either way the
            # MedSigLIP embedding at row i corresponds to bank row i — outputs unchanged.
            n = len(self.paths)
            if "image_paths" in d.files:
                ms_paths = [str(p) for p in d["image_paths"]]
                assert ms_paths == self.paths, "MedSigLIP row order != XRV bank row order"
            else:
                assert d["embeddings"].shape[0] == n, "MedSigLIP bank size != XRV bank size"
            assert int(d["row_ids"].sum()) == int(np.arange(n).sum()), "row_id checksum mismatch"
            self.cuvs_ms = build_retriever(prefer=self._retrieval_prefer, allow_fallback=self._retrieval_allow_fallback).build(
                d["embeddings"].astype(np.float32))
            self.ms_backend = self.cuvs_ms.backend
            self._ms_model_path = ms_model_path
            self._ms_encoder = None
            self.ms_ready = True

    # -- lazy MedSigLIP encoder (only when config C is used) --------------
    def _medsiglip(self):
        if self._ms_encoder is None:
            from gtc_demo.models.medsiglip_encoder import MedSigLIPEncoder
            self._ms_encoder = MedSigLIPEncoder(self._ms_model_path, dtype=self._ms_dtype,
                                                device=self.device)
        return self._ms_encoder

    # -- validated XRV image embedding ------------------------------------
    @torch.inference_mode()
    def encode_xrv(self, images: torch.Tensor) -> torch.Tensor:
        return self.model.encode_image(images.to(self.device))   # (B,256)

    # -- retrieval (encoder swap; row IDs are the shared space) -----------
    @torch.inference_mode()
    def retrieve(self, config: str, xrv_img_emb=None, images_for_ms=None,
                 topk=None) -> RetrievalOut:
        k = int(topk or self.topk)
        enc = RETRIEVAL_CONFIGS[config]["encoder"]
        back = RETRIEVAL_CONFIGS[config]["backend"]
        if config == "A":
            idx, sims = [], []
            for i in range(xrv_img_emb.shape[0]):
                res = self.membank.retrieve(xrv_img_emb[i], mode="similarity_weighted",
                                            topk=k)
                idx.append(res.topk_indices.cpu().numpy())
                sims.append(res.topk_sims.cpu().numpy())
            return RetrievalOut(np.stack(idx).astype(np.int64),
                                np.stack(sims).astype(np.float32), "validated", enc)
        if config == "B":
            r = self.cuvs_xrv.search(
                xrv_img_emb.detach().cpu().numpy().astype(np.float32), k)
            return RetrievalOut(r.indices, r.scores, self.xrv_backend, enc)
        if config == "C":
            ms_emb = self._medsiglip().encode_batch(images_for_ms)  # (B,1152) raw
            r = self.cuvs_ms.search(ms_emb.astype(np.float32), k)
            return RetrievalOut(r.indices, r.scores, self.ms_backend, enc)
        raise ValueError(config)

    # -- pseudo-report (validated aggregation formula) --------------------
    @torch.inference_mode()
    def build_pseudo(self, ro: RetrievalOut) -> torch.Tensor:
        out = torch.zeros(ro.idx.shape[0], self.text_bank.shape[1], device=self.device)
        for i in range(ro.idx.shape[0]):
            idx = torch.as_tensor(ro.idx[i], device=self.device)
            sims = torch.as_tensor(ro.sims[i], device=self.device).float()
            out[i] = self._agg(self.text_bank, idx, sims,
                               weighting=self.weighting, temperature=self.temperature)
        return out

    # -- reliability components (validated formulas) ----------------------
    def reliability_components(self, ro: RetrievalOut) -> dict:
        B, k = ro.idx.shape
        agree = np.zeros(B, np.float32); sim_max = np.zeros(B, np.float32)
        sim_margin = np.zeros(B, np.float32)
        lab = self.labels.numpy()
        for i in range(B):
            s = ro.sims[i]
            sim_max[i] = float(s[0]); sim_margin[i] = float(s[0] - s[-1]) if k > 1 else 0.0
            p = lab[ro.idx[i]].mean(0)
            agree[i] = float(np.mean(np.maximum(p, 1 - p)))
        return {"agree": agree, "sim_max": sim_max, "sim_margin": sim_margin}

    # -- classifier (validated) ------------------------------------------
    @torch.inference_mode()
    def classify(self, xrv_img_emb, pseudo=None):
        B = xrv_img_emb.shape[0]
        if pseudo is None:
            zero = torch.zeros(B, self.model.txt_dim, device=self.device)
            absent = torch.zeros(B, dtype=torch.bool, device=self.device)
            return self.model.classify_from_embeddings(xrv_img_emb, zero, absent)
        present = torch.ones(B, dtype=torch.bool, device=self.device)
        return self.model.classify_from_embeddings(xrv_img_emb, pseudo, present)

    # -- neighbour evidence for display / traceability -------------------
    def neighbors_detail(self, ro: RetrievalOut, i: int) -> list:
        out = []
        for rank, (j, s) in enumerate(zip(ro.idx[i], ro.sims[i]), 1):
            j = int(j)
            out.append({
                "rank": rank, "row_id": j,
                "study_id": os.path.basename(self.paths[j]).split("_")[0],
                "image_path": self.paths[j], "similarity": round(float(s), 4),
                "labels": self.labels[j].int().tolist(),
                "report_excerpt": (self.reports[j] or "")[:220],
                "report_embedding_ref": f"text_bank[{j}]",
            })
        return out

    # -- single-image predict (for the future GUI) -----------------------
    @torch.inference_mode()
    def predict(self, image, retrieval_encoder="medsiglip",
                retrieval_backend="cuvs", topk=None, gate=None) -> dict:
        from PIL import Image
        from utils.transforms import build_transforms
        cfgmap = {("xrv", "validated"): "A", ("xrv", "cuvs"): "B",
                  ("medsiglip", "cuvs"): "C"}
        config = cfgmap[(retrieval_encoder, retrieval_backend)]
        pil = image if isinstance(image, Image.Image) else Image.open(image)
        tf = build_transforms(int(self.cfg.dataset.get("image_size", 224)), train=False)
        x = tf(pil.convert("L")).unsqueeze(0)
        img_emb = self.encode_xrv(x)
        ro = self.retrieve(config, xrv_img_emb=img_emb, images_for_ms=[pil], topk=topk)
        pseudo = self.build_pseudo(ro)
        rc = self.reliability_components(ro)
        p_img = torch.sigmoid(self.classify(img_emb)).cpu().numpy()[0]
        p_pse = torch.sigmoid(self.classify(img_emb, pseudo)).cpu().numpy()[0]
        alpha = None; p_soft = None
        if gate is not None:
            feats = np.array([[rc["sim_max"][0], rc["sim_margin"][0], rc["agree"][0]]])
            alpha = float(gate.predict_proba(feats)[:, 1][0])
            p_soft = ((1 - alpha) * p_img + alpha * p_pse).tolist()
        return {
            "retrieval_encoder": retrieval_encoder, "retrieval_backend": ro.backend,
            "config": config, "findings": self.findings,
            "neighbors": self.neighbors_detail(ro, 0),
            "neighbor_agreement": float(rc["agree"][0]),
            "retrieval_confidence_sim_max": float(rc["sim_max"][0]),
            "sim_margin": float(rc["sim_margin"][0]),
            "image_only_probs": p_img.tolist(),
            "medirely_probs": p_pse.tolist(),
            "reliability_alpha": alpha, "soft_gated_probs": p_soft,
        }
