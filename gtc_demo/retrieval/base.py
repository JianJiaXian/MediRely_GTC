"""Retrieval abstraction shared by all GTC backends.

Design goals
------------
* Minimal, explicit interface: ``build(embeddings)`` then ``search(queries, k)``.
* Every result reports which backend produced it (no silent fallbacks).
* Cosine-equivalent behaviour that MATCHES the validated MediRely path in
  ``utils/retrieval.py``.

Cosine equivalence
------------------
The validated retriever (``utils.retrieval.MemoryBankRetriever``) computes
similarity with ``cosine_similarity_matrix`` = ``l2norm(q) @ l2norm(bank).T``
using ``F.normalize(..., p=2, eps=1e-8)`` at query time (the bank is stored
UN-normalized). Cosine similarity is invariant to *when* the L2 normalization is
applied, so we may pre-normalize the bank once at build time and use an inner
product at search time; the result is identical up to floating point. We
replicate the same ``eps=1e-8`` used by torch's ``F.normalize`` so the two paths
agree to fp tolerance.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Matches torch.nn.functional.normalize default eps used in utils/retrieval.py.
L2_EPS = 1e-8


def l2_normalize(x: np.ndarray, eps: float = L2_EPS) -> np.ndarray:
    """Row-wise L2 normalization matching ``F.normalize(x, p=2, eps=eps)``.

    torch divides by ``max(||x||, eps)`` (not ``||x|| + eps``); we do the same so
    the normalized vectors are bit-for-bit comparable with the validated path.
    """
    x = np.asarray(x, dtype=np.float32)
    norm = np.linalg.norm(x, ord=2, axis=-1, keepdims=True)
    denom = np.maximum(norm, eps)
    return (x / denom).astype(np.float32)


@dataclass
class RetrievalResult:
    """Container returned by every backend's ``search``.

    indices : (Q, k) int64  row IDs into the memory bank (original ordering)
    scores  : (Q, k) float32 similarity scores (cosine / inner product), sorted
              descending per row
    backend : the backend that ACTUALLY produced this result
    """
    indices: np.ndarray
    scores: np.ndarray
    backend: str

    def as_dict(self) -> dict:
        return {
            "indices": self.indices,
            "scores": self.scores,
            "backend": self.backend,
        }


class BaseRetriever:
    """Minimal retrieval interface.

    Subclasses must set ``self.backend`` to the backend actually in use and
    implement ``build`` and ``search``. ``metric`` is documented as
    ``inner_product`` on L2-normalized vectors == cosine similarity.
    """

    backend: str = "base"
    metric: str = "inner_product"

    def build(self, embeddings: np.ndarray) -> "BaseRetriever":
        raise NotImplementedError

    def search(self, queries: np.ndarray, k: int) -> RetrievalResult:
        raise NotImplementedError

    # shared helpers ----------------------------------------------------
    @property
    def num_vectors(self) -> int:
        raise NotImplementedError

    @property
    def dim(self) -> int:
        raise NotImplementedError
