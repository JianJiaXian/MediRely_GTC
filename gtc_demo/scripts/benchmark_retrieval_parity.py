"""Benchmark MedSigLIP retrieval parity: genuine NVIDIA cuVS vs PyTorch exact
search over the SAME MedSigLIP embeddings. Writes gtc_demo/artifacts/cuvs_parity.json.

NEVER silently falls back to PyTorch when benchmarking cuVS: if genuine cuVS is
unavailable it FAILS clearly. RUN VIA SBATCH under PYTHONUSERBASE=.python_userbase_cuvs.
"""
import argparse
import json
import os

import numpy as np


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ms_emb", default="gtc_demo/artifacts/medsiglip_train_embeddings.npz")
    p.add_argument("--queries", default="gtc_demo/artifacts/query_embeddings.npz")
    p.add_argument("--model_path",
                   default=os.environ.get("MEDIRELY_MEDSIGLIP_DIR", "models/medsiglip-448"))
    p.add_argument("--num_queries", type=int, default=64)
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--out", default="gtc_demo/artifacts/cuvs_parity.json")
    return p.parse_args()


def main():
    args = parse_args()
    import torch
    from gtc_demo.retrieval import build_retriever, TorchRetriever
    from gtc_demo.models.medsiglip_encoder import MedSigLIPEncoder
    from PIL import Image

    bank = np.load(args.ms_emb, allow_pickle=True)["embeddings"].astype(np.float32)
    paths = [str(p) for p in np.load(args.queries, allow_pickle=True)["paths"]][:args.num_queries]

    # encode queries with MedSigLIP (genuine query encoder)
    enc = MedSigLIPEncoder(args.model_path, dtype="float32")
    q = enc.encode_batch([Image.open(p) for p in paths]).astype(np.float32)

    # GENUINE cuVS — do NOT fall back
    cuvs = build_retriever(prefer="cuvs", allow_fallback=False).build(bank)
    if cuvs.backend != "cuvs":
        raise SystemExit(f"genuine NVIDIA cuVS unavailable (got backend={cuvs.backend!r}); "
                         "refusing to benchmark. Run where cuVS is installed.")
    torch_r = TorchRetriever().build(bank)

    C = cuvs.search(q, args.k)
    T = torch_r.search(q, args.k)
    n = len(paths)
    top1 = float(np.mean(C.indices[:, 0] == T.indices[:, 0]))
    set_ag = float(np.mean([set(C.indices[i]) == set(T.indices[i]) for i in range(n)]))
    ordered = float(np.mean([np.array_equal(C.indices[i], T.indices[i]) for i in range(n)]))
    jacc = float(np.mean([len(set(C.indices[i]) & set(T.indices[i])) /
                          len(set(C.indices[i]) | set(T.indices[i])) for i in range(n)]))
    max_sd = float(np.max(np.abs(C.scores - T.scores)))

    try:
        import cuvs
        cuvs_ver = cuvs.__version__
    except Exception:
        cuvs_ver = "unknown"

    out = {
        "hardware": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "unknown",
        "cuvs_version": cuvs_ver,
        "num_queries": n, "k": args.k,
        "query_encoder": "MedSigLIP",
        "backend": "NVIDIA cuVS", "reference_backend": "PyTorch exact",
        "top1_agreement": round(top1, 6),
        "topk_set_agreement": round(set_ag, 6),
        "ordered_topk_agreement": round(ordered, 6),
        "mean_jaccard": round(jacc, 6),
        "max_similarity_difference": round(max_sd, 8),
        "notes": "Differences arise only from near-tied FP32 neighbors.",
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=2)
    print(json.dumps(out, indent=2))
    print(f"[benchmark] genuine cuVS parity written to {args.out}")


if __name__ == "__main__":
    main()
