"""Similarity-aware retrieval for the pseudo-modality bridge.

Given a query image embedding and a memory bank of (image_embedding,
text_embedding) pairs from the training set, this module retrieves the
top-k nearest neighbours and aggregates their text embeddings into a
pseudo-report embedding.

All operations are training-free and run on whatever device the tensors
live on (CPU or GPU).
"""
from dataclasses import dataclass, field
from typing import List

import torch
import torch.nn.functional as F


@dataclass
class RetrievalResult:
    """Container for one query's retrieval outcome."""
    pseudo_text: torch.Tensor          # (D,) aggregated pseudo-report embedding
    topk_indices: torch.Tensor         # (k,) indices into the memory bank
    topk_sims: torch.Tensor            # (k,) similarity scores
    max_sim: float                     # highest similarity (used for fallback)
    used_fallback: bool = False        # whether the fallback path was taken
    fallback_type: str = "none"
    meta: dict = field(default_factory=dict)


def _l2norm(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    return F.normalize(x, p=2, dim=dim, eps=1e-8)


def cosine_similarity_matrix(query: torch.Tensor, bank: torch.Tensor) -> torch.Tensor:
    """Cosine similarity between query (Q, D) and bank (N, D) -> (Q, N)."""
    return _l2norm(query) @ _l2norm(bank).t()


def euclidean_similarity_matrix(query: torch.Tensor, bank: torch.Tensor) -> torch.Tensor:
    """Negative L2 distance turned into a similarity (Q, N).

    We return ``-distance`` so that larger == more similar, consistent with the
    cosine path. For softmax weighting this is converted appropriately.
    """
    dist = torch.cdist(query, bank, p=2)  # (Q, N)
    return -dist


def aggregate_pseudo_text(text_bank: torch.Tensor,
                          topk_idx: torch.Tensor,
                          topk_sims: torch.Tensor,
                          weighting: str = "similarity",
                          temperature: float = 0.1) -> torch.Tensor:
    """Aggregate the retrieved text embeddings into one pseudo-report embedding.

    weighting:
      ``similarity`` -> softmax(sim / T) weighted average
      ``average``    -> uniform mean of the k retrieved embeddings
    """
    retrieved = text_bank[topk_idx]  # (k, D)
    if weighting == "average":
        return retrieved.mean(dim=0)
    weights = torch.softmax(topk_sims / max(temperature, 1e-6), dim=0)  # (k,)
    return (weights.unsqueeze(1) * retrieved).sum(dim=0)


class MemoryBankRetriever:
    """Wraps a memory bank and exposes the retrieval modes used in the paper.

    Modes (``mode`` argument of :meth:`retrieve`):
      top1, top3, top5      -> top-k similarity-weighted pseudo-report
      similarity_weighted   -> top-k (cfg) similarity-weighted pseudo-report
      mean_report           -> global mean text embedding (ignores the query)
      random_report         -> a random training report embedding
      zero_text             -> all-zeros pseudo text (image+missing baseline)
      threshold_fallback    -> top-k, but fall back when max_sim < threshold
    """

    def __init__(self, image_bank: torch.Tensor, text_bank: torch.Tensor,
                 metric: str = "cosine", weighting: str = "similarity",
                 temperature: float = 0.1, normalize: bool = True,
                 device: str = "cpu", seed: int = 42):
        self.device = device
        self.metric = metric
        self.weighting = weighting
        self.temperature = temperature
        self.normalize = normalize
        self.image_bank = image_bank.to(device).float()
        self.text_bank = text_bank.to(device).float()
        if normalize:
            self.image_bank_n = _l2norm(self.image_bank)
        else:
            self.image_bank_n = self.image_bank
        self._mean_text = self.text_bank.mean(dim=0)
        self._zero_text = torch.zeros(self.text_bank.shape[1], device=device)
        self._rng = torch.Generator(device="cpu").manual_seed(seed)

    @property
    def num_items(self) -> int:
        return self.image_bank.shape[0]

    @property
    def text_dim(self) -> int:
        return self.text_bank.shape[1]

    def _similarities(self, query: torch.Tensor) -> torch.Tensor:
        query = query.to(self.device).float()
        if self.metric == "euclidean":
            return euclidean_similarity_matrix(query, self.image_bank)[0]
        return cosine_similarity_matrix(query, self.image_bank)[0]

    def retrieve(self, query_emb: torch.Tensor, mode: str = "similarity_weighted",
                 topk: int = 5, fallback: str = "none",
                 similarity_threshold: float = 0.5,
                 image_emb_for_fallback: torch.Tensor = None) -> RetrievalResult:
        """Retrieve a pseudo-report embedding for a single query embedding.

        ``query_emb`` is a (D,) or (1, D) image embedding.
        """
        if query_emb.dim() == 1:
            query_emb = query_emb.unsqueeze(0)

        # Static modes that do not depend on similarity ranking.
        if mode == "mean_report":
            return RetrievalResult(self._mean_text.clone(),
                                   torch.empty(0), torch.empty(0), 0.0,
                                   meta={"mode": mode})
        if mode == "zero_text":
            return RetrievalResult(self._zero_text.clone(),
                                   torch.empty(0), torch.empty(0), 0.0,
                                   meta={"mode": mode})
        if mode == "random_report":
            idx = torch.randint(0, self.num_items, (1,), generator=self._rng).item()
            return RetrievalResult(self.text_bank[idx].clone(),
                                   torch.tensor([idx]), torch.tensor([0.0]), 0.0,
                                   meta={"mode": mode})

        # Similarity-based modes.
        sims = self._similarities(query_emb)  # (N,)
        k = {"top1": 1, "top3": 3, "top5": 5}.get(mode, topk)
        k = min(k, self.num_items)
        topk_sims, topk_idx = torch.topk(sims, k)
        max_sim = float(topk_sims[0].item())

        weighting = self.weighting
        pseudo = aggregate_pseudo_text(self.text_bank, topk_idx, topk_sims,
                                       weighting=weighting,
                                       temperature=self.temperature)

        used_fallback = False
        fb_type = "none"
        if fallback != "none" and max_sim < similarity_threshold:
            used_fallback = True
            fb_type = fallback
            if fallback == "mean":
                pseudo = self._mean_text.clone()
            elif fallback in ("image_only", "zero"):
                pseudo = self._zero_text.clone()
            elif fallback == "threshold":
                # "threshold" default behaviour: drop to the zero/image-only path
                pseudo = self._zero_text.clone()

        return RetrievalResult(pseudo, topk_idx.cpu(), topk_sims.cpu(), max_sim,
                               used_fallback=used_fallback, fallback_type=fb_type,
                               meta={"mode": mode, "k": k})

    def retrieve_batch(self, query_embs: torch.Tensor, mode: str = "similarity_weighted",
                       topk: int = 5, fallback: str = "none",
                       similarity_threshold: float = 0.5) -> List[RetrievalResult]:
        """Vectorised-ish convenience wrapper (loops per query for clarity)."""
        return [self.retrieve(q, mode=mode, topk=topk, fallback=fallback,
                              similarity_threshold=similarity_threshold)
                for q in query_embs]


def kmeans_torch(x: torch.Tensor, n_clusters: int, n_iter: int = 25,
                 seed: int = 42):
    """Tiny dependency-free k-means (Lloyd) on row vectors x (N, D).

    Returns (assignments (N,), centroids (k, D)). Used for the grouping-based
    pseudo-modality variant so we avoid a hard sklearn dependency at inference.
    """
    N = x.shape[0]
    n_clusters = min(n_clusters, N)
    g = torch.Generator(device="cpu").manual_seed(seed)
    init = torch.randperm(N, generator=g)[:n_clusters]
    centroids = x[init].clone()
    assign = torch.zeros(N, dtype=torch.long, device=x.device)
    for _ in range(n_iter):
        d = torch.cdist(x, centroids)          # (N, k)
        new_assign = d.argmin(dim=1)
        if torch.equal(new_assign, assign):
            assign = new_assign
            break
        assign = new_assign
        for c in range(n_clusters):
            m = assign == c
            if m.any():
                centroids[c] = x[m].mean(dim=0)
    return assign, centroids


class GroupPrototypeRetriever:
    """Grouping-based pseudo-modality retrieval (advisor-suggested variant).

    Instead of (or in addition to) instance-level top-$k$ retrieval, we partition
    the memory bank into groups and represent each group by a prototype:
      * a centroid IMAGE embedding (for assigning a query to a group), and
      * a prototype TEXT embedding (the group's mean report embedding), used as
        the pseudo-report.

    Two grouping strategies:
      * ``kmeans``  : cluster training image embeddings into ``n_groups``.
      * ``label``   : group by exact multi-label vector (clinically meaningful
                      prototypes; requires labels at build time only).

    At inference the query is assigned to its nearest group centroid by cosine
    similarity and inherits that group's text prototype. This is more robust to
    noisy single neighbours and exposes a confidence (distance to the centroid)
    for the same similarity-aware fallback.
    """

    def __init__(self, image_bank: torch.Tensor, text_bank: torch.Tensor,
                 labels: torch.Tensor = None, strategy: str = "kmeans",
                 n_groups: int = 16, normalize: bool = True,
                 device: str = "cpu", seed: int = 42):
        self.device = device
        self.normalize = normalize
        image_bank = image_bank.to(device).float()
        text_bank = text_bank.to(device).float()
        if strategy == "label" and labels is not None:
            assign, n_groups = self._label_groups(labels.to(device))
        else:
            assign, _ = kmeans_torch(image_bank.cpu(), n_groups, seed=seed)
            assign = assign.to(device)
        self.n_groups = int(assign.max().item()) + 1
        # group prototypes
        img_protos, txt_protos = [], []
        for c in range(self.n_groups):
            m = assign == c
            if m.any():
                img_protos.append(image_bank[m].mean(0))
                txt_protos.append(text_bank[m].mean(0))
            else:  # empty group safeguard
                img_protos.append(torch.zeros(image_bank.shape[1], device=device))
                txt_protos.append(torch.zeros(text_bank.shape[1], device=device))
        self.img_protos = torch.stack(img_protos)     # (G, Di)
        self.txt_protos = torch.stack(txt_protos)     # (G, Dt)
        self.assign = assign

    @staticmethod
    def _label_groups(labels):
        labels = (labels > 0.5).long()
        uniq, inv = torch.unique(labels, dim=0, return_inverse=True)
        return inv, uniq.shape[0]

    def retrieve(self, query_emb: torch.Tensor, fallback: str = "none",
                 similarity_threshold: float = 0.5) -> RetrievalResult:
        if query_emb.dim() == 1:
            query_emb = query_emb.unsqueeze(0)
        sims = cosine_similarity_matrix(query_emb, self.img_protos)[0]  # (G,)
        best = int(sims.argmax().item())
        max_sim = float(sims[best].item())
        pseudo = self.txt_protos[best].clone()
        used_fb = False
        fb_type = "none"
        if fallback != "none" and max_sim < similarity_threshold:
            used_fb, fb_type = True, fallback
            if fallback == "mean":
                pseudo = self.txt_protos.mean(0)
            else:  # zero / image_only
                pseudo = torch.zeros_like(pseudo)
        return RetrievalResult(pseudo, torch.tensor([best]),
                               torch.tensor([max_sim]), max_sim,
                               used_fallback=used_fb, fallback_type=fb_type,
                               meta={"mode": "group_prototype",
                                     "group": best, "n_groups": self.n_groups})
