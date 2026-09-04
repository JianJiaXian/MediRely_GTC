"""Clean-room torch brute-force retriever (cosine via inner product).

This mirrors the math of the validated ``utils.retrieval.MemoryBankRetriever``
cosine path exactly (L2-normalize both sides with eps=1e-8, inner product, top-k)
but is a self-contained backend so the GTC layer never imports/edits the research
code at run time. It runs on GPU when available, else CPU.

It is used both as the ``torch`` fallback in the backend hierarchy and as an
independent cross-check target in the parity job.
"""
from __future__ import annotations

import numpy as np

from gtc_demo.retrieval.base import BaseRetriever, RetrievalResult, L2_EPS


class TorchRetriever(BaseRetriever):
    backend = "torch"
    metric = "inner_product"  # on L2-normalized rows == cosine

    def __init__(self, device: str = None):
        import torch
        self._torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._bank = None          # (N, D) normalized, on device
        self._n = 0
        self._d = 0

    def _l2norm(self, t):
        # torch F.normalize semantics: divide by max(||x||, eps)
        return self._torch.nn.functional.normalize(t, p=2, dim=-1, eps=L2_EPS)

    def build(self, embeddings: np.ndarray) -> "TorchRetriever":
        torch = self._torch
        emb = torch.as_tensor(np.asarray(embeddings, dtype=np.float32),
                              device=self.device)
        self._n, self._d = int(emb.shape[0]), int(emb.shape[1])
        self._bank = self._l2norm(emb).contiguous()
        return self

    def search(self, queries: np.ndarray, k: int) -> RetrievalResult:
        torch = self._torch
        assert self._bank is not None, "call build() first"
        q = torch.as_tensor(np.asarray(queries, dtype=np.float32),
                            device=self.device)
        if q.dim() == 1:
            q = q.unsqueeze(0)
        q = self._l2norm(q)
        k = int(min(k, self._n))
        sims = q @ self._bank.t()                     # (Q, N) cosine
        scores, idx = torch.topk(sims, k, dim=1)      # descending
        return RetrievalResult(
            indices=idx.detach().cpu().numpy().astype(np.int64),
            scores=scores.detach().cpu().numpy().astype(np.float32),
            backend=self.backend,
        )

    @property
    def num_vectors(self) -> int:
        return self._n

    @property
    def dim(self) -> int:
        return self._d
