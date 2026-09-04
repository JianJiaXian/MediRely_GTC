"""Local-GPU self-test (runs on any CUDA GPU). Simulates the native-Windows path
by FORCING the PyTorch-CUDA retrieval fallback, then measures VRAM + latency and
checks retrieval / prediction / reliability parity against the genuine-cuVS path.

On the H200 this validates the code + produces parity numbers; the VRAM/latency
figures are a proxy (must be confirmed on the 6 GB RTX 2060). RUN VIA SBATCH.
"""
import json
import time

import numpy as np
import torch

CASES = ["strong_recovery", "coherent_evidence", "conflicting_evidence"]


def _mb(x):
    return round(x / 1e6, 1)


def _neighbors(pred):
    return [n["row_id"] for n in pred["neighbors"]], \
           np.array([n["similarity"] for n in pred["neighbors"]], dtype=np.float64)


def main():
    from gtc_demo.pipeline import MediRelyGTC
    from gtc_demo.app.adapters import LogisticGate
    gate = LogisticGate.from_artifact()
    src = json.load(open("gtc_demo/artifacts/gallery_sources.json", encoding="utf-8"))
    imgs = {k: src[k]["image_path"] for k in CASES}

    dev = torch.cuda.get_device_name(0)
    report = {"gpu": dev, "torch": torch.__version__, "cuda": torch.version.cuda,
              "note": "VRAM/latency measured on this GPU as a PROXY; confirm on RTX 2060 6GB."}

    # ---------- VRAM: LOCAL path (forced PyTorch CUDA retrieval) ----------
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    vram = {"startup_MB": _mb(torch.cuda.memory_allocated())}
    gtc = MediRelyGTC(topk=10, device="cuda", retrieval_prefer="torch", ms_dtype="float32")
    vram["after_models_load_MB"] = _mb(torch.cuda.memory_allocated())
    report["local_retrieval_backend"] = gtc.ms_backend      # must be 'torch'

    p0 = gtc.predict(imgs[CASES[0]], retrieval_encoder="medsiglip",
                     retrieval_backend="cuvs", gate=gate)     # config C; warms MedSigLIP
    torch.cuda.synchronize()
    vram["after_medsiglip_infer_MB"] = _mb(torch.cuda.memory_allocated())
    vram["peak_end_to_end_MB"] = _mb(torch.cuda.max_memory_allocated())
    report["vram"] = vram
    report["fits_6GB"] = vram["peak_end_to_end_MB"] < 6000

    # ---------- latency breakdown (warm) ----------
    from PIL import Image
    from utils.transforms import build_transforms
    tf = build_transforms(224, train=False)
    pth = imgs[CASES[0]]
    for _ in range(3):
        gtc.predict(pth, retrieval_encoder="medsiglip", retrieval_backend="cuvs", gate=gate)
    torch.cuda.synchronize()

    def sync():
        torch.cuda.synchronize()
    lat = {}
    t0 = time.perf_counter(); pil = Image.open(pth); x = tf(pil.convert("L")).unsqueeze(0); sync()
    lat["preprocess_ms"] = round((time.perf_counter() - t0) * 1000, 2)
    t0 = time.perf_counter(); emb = gtc.encode_xrv(x); sync()
    lat["xrv_encode_ms"] = round((time.perf_counter() - t0) * 1000, 2)
    t0 = time.perf_counter(); ro = gtc.retrieve("C", images_for_ms=[pil], topk=10); sync()
    lat["medsiglip_encode+retrieve_ms"] = round((time.perf_counter() - t0) * 1000, 2)
    t0 = time.perf_counter(); ps = gtc.build_pseudo(ro); sync()
    lat["pseudo_aggregate_ms"] = round((time.perf_counter() - t0) * 1000, 2)
    t0 = time.perf_counter(); gtc.reliability_components(ro); sync()
    lat["reliability_ms"] = round((time.perf_counter() - t0) * 1000, 2)
    t0 = time.perf_counter(); gtc.classify(emb, ps); sync()
    lat["classifier_ms"] = round((time.perf_counter() - t0) * 1000, 2)
    runs = []
    for _ in range(5):
        t0 = time.perf_counter()
        gtc.predict(pth, retrieval_encoder="medsiglip", retrieval_backend="cuvs", gate=gate)
        sync(); runs.append((time.perf_counter() - t0) * 1000)
    lat["total_end_to_end_ms_mean"] = round(float(np.mean(runs)), 2)
    report["latency"] = lat

    # local (torch) predictions for all cases
    local = {}
    for k in CASES:
        pr = gtc.predict(imgs[k], retrieval_encoder="medsiglip",
                         retrieval_backend="cuvs", gate=gate)
        idx, sims = _neighbors(pr)
        local[k] = {"idx": idx, "sims": sims, "alpha": float(pr["reliability_alpha"]),
                    "p_img": np.array(pr["image_only_probs"]),
                    "p_med": np.array(pr["medirely_probs"])}
    del gtc
    torch.cuda.empty_cache()

    # ---------- genuine cuVS reference ----------
    gtc_c = MediRelyGTC(topk=10, device="cuda", retrieval_prefer="cuvs", ms_dtype="float32")
    report["reference_backend"] = gtc_c.ms_backend          # 'cuvs'
    parity = {}
    for k in CASES:
        pr = gtc_c.predict(imgs[k], retrieval_encoder="medsiglip",
                           retrieval_backend="cuvs", gate=gate)
        ridx, rsims = _neighbors(pr)
        lidx, lsims = local[k]["idx"], local[k]["sims"]
        setov = len(set(lidx) & set(ridx)) / len(set(lidx) | set(ridx))
        parity[k] = {
            "top1_agree": int(lidx[0] == ridx[0]),
            "topk_set_agree": round(setov, 4),
            "topk_ordered_agree": int(lidx == ridx),
            "max_sim_diff": round(float(np.max(np.abs(lsims - rsims))), 6),
            "reliability_alpha_diff": round(abs(local[k]["alpha"] - float(pr["reliability_alpha"])), 6),
            "pred_prob_max_diff": round(float(np.max(np.abs(local[k]["p_med"] - np.array(pr["medirely_probs"])))), 6),
        }
    report["parity_torch_vs_cuvs"] = parity
    report["parity_pass"] = all(
        v["top1_agree"] == 1 and v["topk_set_agree"] >= 0.99 and
        v["reliability_alpha_diff"] <= 0.02 and v["pred_prob_max_diff"] <= 0.02
        for v in parity.values())

    import os
    os.makedirs("artifacts_local", exist_ok=True)
    json.dump(report, open("artifacts_local/local_gpu_selftest.json", "w"), indent=2)
    print(json.dumps(report, indent=2))
    print(f"[selftest] local backend={report['local_retrieval_backend']} "
          f"peakVRAM={report['vram']['peak_end_to_end_MB']}MB fits6GB={report['fits_6GB']} "
          f"parity_pass={report['parity_pass']}")


if __name__ == "__main__":
    main()
