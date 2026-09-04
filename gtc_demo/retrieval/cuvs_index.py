"""NVIDIA cuVS brute-force retriever with explicit backend reporting.

``CuVSRetriever`` prefers genuine NVIDIA **cuVS** GPU brute-force search. If cuVS
is not importable in the current environment it falls back, in order, to
FAISS (GPU then CPU) and finally the clean-room ``TorchRetriever``. The backend
that is ACTUALLY used is recorded on ``self.backend`` and stamped into every
``RetrievalResult`` -- we never report ``cuvs`` while running something else.

All backends use the same cosine-equivalent formulation: vectors are
L2-normalized (eps=1e-8, matching utils/retrieval.py) and compared by inner
product. Exact (brute-force) search is used so results are directly comparable
to the validated torch reference.
"""
from __future__ import annotations

import time

import numpy as np

from gtc_demo.retrieval.base import BaseRetriever, RetrievalResult, l2_normalize


# ---------------------------------------------------------------------------
# backend availability probes (import lazily; never at module import time)
# ---------------------------------------------------------------------------
def _try_import_cuvs():
    try:
        from cuvs.neighbors import brute_force  # noqa: F401
        import cuvs  # noqa: F401
        return brute_force
    except Exception:
        return None


def _try_import_raft():
    """RAFT brute-force (NVIDIA) -- cuVS's predecessor, ships in the container."""
    try:
        from pylibraft.neighbors.brute_force import knn  # noqa: F401
        return knn
    except Exception:
        return None


def _try_import_faiss():
    try:
        import faiss
        return faiss
    except Exception:
        return None


def _cosine_from_sqeuclidean(dist):
    """For L2-normalized rows: ||a-b||^2 = 2 - 2 cos  =>  cos = 1 - dist/2.

    knn/brute-force return the k SMALLEST squared-euclidean distances, which are
    exactly the k LARGEST cosine similarities -- unambiguous ordering, no
    metric-sign guesswork. Cosine is recovered exactly.
    """
    return (1.0 - np.asarray(dist, dtype=np.float32) / 2.0).astype(np.float32)


# ---------------------------------------------------------------------------
# concrete implementations
# ---------------------------------------------------------------------------
class _CuvsImpl:
    """Genuine NVIDIA cuVS brute-force. Uses sqeuclidean on L2-normalized rows
    so the k-nearest are exactly the k-highest-cosine (order unambiguous)."""

    backend = "cuvs"

    def __init__(self, brute_force):
        self._bf = brute_force
        self._index = None
        self._bank_dev = None
        self._n = 0
        self._d = 0

    def build(self, norm_bank: np.ndarray):
        import cupy as cp
        self._n, self._d = norm_bank.shape
        self._bank_dev = cp.asarray(norm_bank, dtype=cp.float32)
        self._index = self._bf.build(self._bank_dev, metric="sqeuclidean")

    def search(self, norm_q: np.ndarray, k: int):
        import cupy as cp
        q = cp.asarray(norm_q, dtype=cp.float32)
        dist, nbr = self._bf.search(self._index, q, k)
        dist = cp.asnumpy(cp.asarray(dist))
        idx = cp.asnumpy(cp.asarray(nbr)).astype(np.int64)
        scores = _cosine_from_sqeuclidean(dist)          # k-nearest already desc-cos
        return idx, scores


class _RaftImpl:
    """NVIDIA RAFT brute-force knn (pylibraft) -- cuVS lineage, ships in the SIF."""

    backend = "raft"

    def __init__(self, knn):
        self._knn = knn
        self._bank_dev = None
        self._n = 0
        self._d = 0

    def build(self, norm_bank: np.ndarray):
        import cupy as cp
        self._n, self._d = norm_bank.shape
        self._bank_dev = cp.ascontiguousarray(cp.asarray(norm_bank, dtype=cp.float32))

    def search(self, norm_q: np.ndarray, k: int):
        import cupy as cp
        q = cp.ascontiguousarray(cp.asarray(norm_q, dtype=cp.float32))
        dist, nbr = self._knn(self._bank_dev, q, k, metric="sqeuclidean")
        dist = cp.asnumpy(cp.asarray(dist))
        idx = cp.asnumpy(cp.asarray(nbr)).astype(np.int64)
        scores = _cosine_from_sqeuclidean(dist)
        return idx, scores


class _FaissImpl:
    def __init__(self, faiss):
        self._faiss = faiss
        self._index = None
        self._n = 0
        self._d = 0
        gpus = 0
        try:
            gpus = faiss.get_num_gpus()
        except Exception:
            gpus = 0
        self._use_gpu = gpus > 0
        self.backend = "faiss-gpu" if self._use_gpu else "faiss-cpu"

    def build(self, norm_bank: np.ndarray):
        self._n, self._d = norm_bank.shape
        index = self._faiss.IndexFlatIP(self._d)   # exact inner product
        if self._use_gpu:
            res = self._faiss.StandardGpuResources()
            index = self._faiss.index_cpu_to_gpu(res, 0, index)
            self._res = res
        index.add(np.ascontiguousarray(norm_bank, dtype=np.float32))
        self._index = index

    def search(self, norm_q: np.ndarray, k: int):
        scores, idx = self._index.search(
            np.ascontiguousarray(norm_q, dtype=np.float32), k)
        return idx.astype(np.int64), scores.astype(np.float32)


# ---------------------------------------------------------------------------
# public retriever
# ---------------------------------------------------------------------------
class CuVSRetriever(BaseRetriever):
    """Prefer cuVS; explicit fallback to FAISS then torch. Reports true backend."""

    metric = "inner_product"

    def __init__(self, allow_fallback: bool = True, device: str = None):
        self.allow_fallback = allow_fallback
        self._device = device
        self._impl = None
        self.backend = "uninitialized"
        self._norm_bank = None
        self._n = 0
        self._d = 0
        self.build_time_s = None

    def _select_impl(self):
        bf = _try_import_cuvs()
        if bf is not None:
            try:
                return _CuvsImpl(bf)
            except Exception:
                pass
        if not self.allow_fallback:
            raise RuntimeError(
                "cuVS unavailable and allow_fallback=False; refusing to run a "
                "non-cuVS backend while labelled cuVS.")
        knn = _try_import_raft()
        if knn is not None:
            try:
                return _RaftImpl(knn)
            except Exception:
                pass
        faiss = _try_import_faiss()
        if faiss is not None:
            return _FaissImpl(faiss)
        # last resort: torch (wrapped to match the impl protocol)
        from gtc_demo.retrieval.torch_retriever import TorchRetriever
        tr = TorchRetriever(device=self._device)

        class _TorchImpl:
            backend = "torch"

            def build(self, norm_bank):
                # TorchRetriever normalizes internally; pass already-normalized
                # rows (idempotent under F.normalize) to keep one code path.
                tr.build(norm_bank)

            def search(self, norm_q, k):
                res = tr.search(norm_q, k)
                return res.indices, res.scores

        return _TorchImpl()

    def build(self, embeddings: np.ndarray) -> "CuVSRetriever":
        emb = np.asarray(embeddings, dtype=np.float32)
        self._n, self._d = emb.shape
        self._norm_bank = l2_normalize(emb)          # cosine -> inner product
        self._impl = self._select_impl()
        self.backend = self._impl.backend
        t0 = time.perf_counter()
        self._impl.build(self._norm_bank)
        self.build_time_s = time.perf_counter() - t0
        return self

    def search(self, queries: np.ndarray, k: int) -> RetrievalResult:
        assert self._impl is not None, "call build() first"
        q = np.asarray(queries, dtype=np.float32)
        if q.ndim == 1:
            q = q[None, :]
        norm_q = l2_normalize(q)
        k = int(min(k, self._n))
        idx, scores = self._impl.search(norm_q, k)
        return RetrievalResult(indices=idx.astype(np.int64),
                               scores=scores.astype(np.float32),
                               backend=self.backend)

    @property
    def num_vectors(self) -> int:
        return self._n

    @property
    def dim(self) -> int:
        return self._d


def build_retriever(prefer: str = "cuvs", allow_fallback: bool = True,
                    device: str = None) -> BaseRetriever:
    """Factory. ``prefer='torch'`` forces TorchRetriever; ``prefer='cuvs'`` uses
    CuVSRetriever (cuVS -> FAISS -> torch, backend reported explicitly)."""
    if prefer == "torch":
        from gtc_demo.retrieval.torch_retriever import TorchRetriever
        return TorchRetriever(device=device)
    return CuVSRetriever(allow_fallback=allow_fallback, device=device)
