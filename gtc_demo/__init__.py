"""GTC demo extension for MediRely (NVIDIA GTC Golden Ticket).

This package is an ISOLATED extension. It never modifies the validated MICCAI
MediRely tree; it only imports from it (datasets/, models/, utils/) read-only.

Step 2 scope: a production-safe NVIDIA cuVS retrieval backend over the existing
validated XRV memory-bank embeddings, verified for Top-K parity against the
validated torch brute-force cosine retrieval in utils/retrieval.py.
"""
__all__ = ["retrieval"]
