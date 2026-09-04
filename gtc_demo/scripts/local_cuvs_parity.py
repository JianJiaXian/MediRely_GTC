"""End-to-end torch-vs-cuVS parity over the 3 verified demo cases.

Runs the FULL MediRely pipeline (MedSigLIP → retrieval → pseudo-context →
reliability → prediction) with genuine NVIDIA cuVS and with PyTorch exact search,
sharing ONE MedSigLIP encoder + one XRV model (memory-safe on 6 GB), so ONLY the
retrieval backend differs. Fails clearly if genuine cuVS is unavailable — never
silently uses torch for the 'cuvs' side.

Writes gtc_demo/artifacts/local_cuvs_end_to_end_parity.json.
Run under WSL2 (medirely_cuvs) or on any host with genuine cuVS. No SLURM needed.
"""
import argparse
import json
import os

import numpy as np


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--out", default="gtc_demo/artifacts/local_cuvs_end_to_end_parity.json")
    return p.parse_args()


def _sig(x):
    return 1.0 / (1.0 + np.exp(-x))


def main():
    args = parse_args()
    import torch
    from PIL import Image
    from utils.transforms import build_transforms
    from gtc_demo.app.local_config import LocalConfig
    from gtc_demo.app.adapters import LogisticGate
    from gtc_demo.pipeline import MediRelyGTC
    from gtc_demo.pipeline.medirely_gtc import RetrievalOut
    from gtc_demo.retrieval import build_retriever

    cfg = LocalConfig()
    gate = LogisticGate.from_artifact(cfg.comparison())
    sources = json.load(open(cfg.gallery_sources(), encoding="utf-8"))

    # ONE pipeline with GENUINE cuVS (strict: raises if cuVS unavailable)
    gtc = MediRelyGTC(config_path=cfg.config_yaml(), checkpoint=cfg.checkpoint(),
                      xrv_bank=cfg.xrv_bank(), ms_emb=cfg.ms_emb(),
                      ms_model_path=cfg.ms_model_path(), topk=args.k, device="cuda",
                      retrieval_prefer="cuvs", ms_dtype="float32",
                      retrieval_allow_fallback=False)
    if gtc.ms_backend != "cuvs":
        raise SystemExit(f"genuine cuVS required but backend={gtc.ms_backend!r}")
    # PyTorch exact retriever over the SAME MedSigLIP bank (shared encoder/model)
    ms_bank = np.load(cfg.ms_emb(), allow_pickle=True)["embeddings"].astype(np.float32)
    torch_ms = build_retriever(prefer="torch").build(ms_bank)

    tf = build_transforms(int(gtc.cfg.dataset.get("image_size", 224)), train=False)

    def run(path, retr):
        pil = Image.open(path)
        x = tf(pil.convert("L")).unsqueeze(0)
        img_emb = gtc.encode_xrv(x)
        ms_q = gtc._medsiglip().encode_batch([pil]).astype(np.float32)
        r = retr.search(ms_q, args.k)
        ro = RetrievalOut(r.indices, r.scores, retr.backend, "medsiglip")
        pseudo = gtc.build_pseudo(ro)
        rc = gtc.reliability_components(ro)
        p_img = _sig(gtc.classify(img_emb).cpu().numpy()[0])
        p_med = _sig(gtc.classify(img_emb, pseudo).cpu().numpy()[0])
        sm, mg, ag = float(rc["sim_max"][0]), float(rc["sim_margin"][0]), float(rc["agree"][0])
        alpha = float(gate.predict_proba([[sm, mg, ag]])[0, 1])
        idx = r.indices[0].tolist()
        return {"idx": idx, "sims": r.scores[0].astype(float),
                "reports": [gtc.reports[i][:200] for i in idx],
                "sim_max": sm, "agree": ag, "alpha": alpha,
                "p_img": p_img, "p_med": p_med,
                "headline": gtc.findings[int(np.argmax(p_med))]}

    cases, max_sim, max_rel, max_pred = [], 0.0, 0.0, 0.0
    all_pass = True
    for key in ("strong_recovery", "coherent_evidence", "conflicting_evidence"):
        path = sources[key]["image_path"]
        c, t = run(path, gtc.cuvs_ms), run(path, torch_ms)
        sim_d = float(np.max(np.abs(c["sims"] - t["sims"])))
        rel_d = max(abs(c["alpha"] - t["alpha"]), abs(c["agree"] - t["agree"]),
                    abs(c["sim_max"] - t["sim_max"]))
        pred_d = float(np.max(np.abs(c["p_med"] - t["p_med"])))
        rec = {
            "case": key,
            "top1_same": c["idx"][0] == t["idx"][0],
            "topk_set_same": set(c["idx"]) == set(t["idx"]),
            "ordered_topk_same": c["idx"] == t["idx"],
            "reports_identical": c["reports"] == t["reports"],
            "max_similarity_diff": round(sim_d, 8),
            "max_reliability_diff": round(rel_d, 8),
            "max_prediction_prob_diff": round(pred_d, 8),
            "headline_same": c["headline"] == t["headline"],
            "cuvs": {"top1": c["idx"][0], "reliability_alpha": round(c["alpha"], 6),
                     "headline": c["headline"], "pred_top": round(float(max(c["p_med"])), 6)},
            "torch": {"top1": t["idx"][0], "reliability_alpha": round(t["alpha"], 6),
                      "headline": t["headline"], "pred_top": round(float(max(t["p_med"])), 6)},
        }
        max_sim, max_rel, max_pred = max(max_sim, sim_d), max(max_rel, rel_d), max(max_pred, pred_d)
        ok = (rec["top1_same"] and rec["topk_set_same"] and rec["reports_identical"]
              and rec["headline_same"] and rel_d <= 1e-4 and pred_d <= 1e-4)
        rec["pass"] = ok
        all_pass = all_pass and ok
        cases.append(rec)
        print(f"[parity:{key}] top1={rec['top1_same']} set={rec['topk_set_same']} "
              f"ordered={rec['ordered_topk_same']} reports={rec['reports_identical']} "
              f"relΔ={rel_d:.2e} predΔ={pred_d:.2e} pass={ok}")

    out = {
        "hardware": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "unknown",
        "backend_a": "NVIDIA cuVS (genuine, brute-force)",
        "backend_b": "PyTorch exact (CUDA)",
        "num_cases": len(cases), "k": args.k,
        "max_similarity_difference": round(max_sim, 8),
        "max_reliability_difference": round(max_rel, 8),
        "max_prediction_probability_difference": round(max_pred, 8),
        "pass": all_pass,
        "cases": cases,
        "notes": "Shared MedSigLIP encoder + XRV model; only the retrieval backend "
                 "differs. Prediction/reliability/pseudo-context logic unchanged.",
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=2)
    print(json.dumps({k: out[k] for k in
                      ("hardware", "num_cases", "k", "max_similarity_difference",
                       "max_reliability_difference", "max_prediction_probability_difference",
                       "pass")}, indent=2))
    print(f"[local_cuvs_parity] wrote {args.out}")


if __name__ == "__main__":
    main()
