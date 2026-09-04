"""Retrieval backends for the GTC demo.

Public API:
    BaseRetriever      - minimal interface (build / search)
    RetrievalResult    - {"indices", "scores", "backend"}
    TorchRetriever     - clean-room torch brute-force (cosine via inner product)
    CuVSRetriever      - NVIDIA cuVS brute-force, with explicit backend reporting
    build_retriever    - factory honouring an explicit preference

IMPORTANT: the selected backend is ALWAYS reported explicitly. A retriever that
falls back (cuVS -> FAISS -> torch) will set ``.backend`` to the backend that was
actually used, and every search result carries that same ``backend`` string.
Nothing here silently claims cuVS while running torch.
"""
from gtc_demo.retrieval.base import BaseRetriever, RetrievalResult, l2_normalize
from gtc_demo.retrieval.torch_retriever import TorchRetriever
from gtc_demo.retrieval.cuvs_index import CuVSRetriever, build_retriever

__all__ = [
    "BaseRetriever",
    "RetrievalResult",
    "l2_normalize",
    "TorchRetriever",
    "CuVSRetriever",
    "build_retriever",
]
