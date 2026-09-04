"""MediRely GTC demo — Gradio application (premium dark navy workstation).

Three modes, SAME polished GUI:

  cluster (default):  python -m gtc_demo.app.app
     H200 contest deployment — REAL pipeline, genuine NVIDIA cuVS. UNCHANGED.

  --local-gpu:        python -m gtc_demo.app.app --local-gpu
     REAL inference on a local Windows NVIDIA GPU (e.g. RTX 2060 6 GB). Uses the
     validated exported artifacts; retrieval backend is genuine cuVS if importable,
     else an explicit PyTorch CUDA exact-search fallback (backend shown truthfully).

  --preview:          python -m gtc_demo.app.app --preview
     CPU-only UI preview from recorded verified outputs; arbitrary uploads do NOT
     run inference. Imports no torch/cuvs/transformers.

Validated MICCAI tree untouched. Cluster mode behaviour is unchanged.
"""
import argparse
import base64
import json
import os
import time

import gradio as gr

from gtc_demo.app import components as C
from gtc_demo.app import tokens as TK

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
HERE = os.path.dirname(os.path.abspath(__file__))
CSS = open(os.path.join(HERE, "styles.css"), encoding="utf-8").read()
ASSETS = os.path.join(os.path.dirname(HERE), "demo_assets")
GALLERY = json.load(open(os.path.join(HERE, "gallery.json"), encoding="utf-8"))
BYKEY = {g["key"]: g for g in GALLERY}

MODE = "cluster"                 # "cluster" | "local" | "preview"
_GTC = None
_GATE = None
_SOURCES = None
_PREVIEW = {}
_IMAGE_ROOT = None
_RUNTIME = {"gpu": None, "vram": None, "backend_label": "NVIDIA cuVS", "is_cuvs": True}


def _file_b64(path):
    with open(path, "rb") as f:
        return "data:image/png;base64," + base64.b64encode(f.read()).decode()


def _valid_image(path):
    try:
        from PIL import Image
        Image.open(path).verify()
        return True
    except Exception:
        return False


def _explain_html(key=None, upload=False):
    if upload:
        return ('<div class="mr-explain">Uploaded chest X-ray — click '
                '<b>Recover with MediRely</b> to run live inference.</div>')
    if key and key in BYKEY:
        g = BYKEY[key]
        return f'<div class="mr-explain"><b>{g["label"]}</b> — {g["explanation"]}</div>'
    return ('<div class="mr-explain">Select a demo case to see why it is useful, '
            'or upload your own chest X-ray.</div>')


# =========================== startup ===================================
def _patch_torch_load_compat():
    """torch<1.13 has no `weights_only` kwarg for torch.load. Our checkpoint/bank
    loaders pass weights_only=False (needed on torch>=2.6). On older torch (e.g.
    1.12 + cu113 on laptop drivers capped at CUDA 11.3) we strip the unsupported
    kwarg so loading still works. No-op on modern torch."""
    import inspect
    import torch
    try:
        supported = "weights_only" in inspect.signature(torch.load).parameters
    except (TypeError, ValueError):
        supported = True
    if not supported and not getattr(torch.load, "_mr_compat", False):
        _orig = torch.load
        def _load(*a, **k):
            k.pop("weights_only", None)
            return _orig(*a, **k)
        _load._mr_compat = True
        torch.load = _load
        print("[compat] torch<1.13 detected — stripped weights_only from torch.load")


def _print_checks(title, checks, extra=""):
    print("=" * 54, f"\n MediRely GTC demo — startup readiness ({title})")
    for k, v in checks.items():
        print(f"  [{'OK' if v else 'FAIL'}] {k}")
    if extra:
        print(extra)
    print("=" * 54)
    if not all(checks.values()):
        raise RuntimeError("startup checks failed; refusing to launch.")


def startup_real():
    """Cluster / H200 REAL mode — genuine NVIDIA cuVS. Unchanged behaviour."""
    global _GTC, _GATE, _SOURCES, _RUNTIME
    _patch_torch_load_compat()
    from gtc_demo.pipeline import MediRelyGTC
    from gtc_demo.app.adapters import LogisticGate
    _SOURCES = json.load(open("gtc_demo/artifacts/gallery_sources.json", encoding="utf-8"))
    _GATE = LogisticGate.from_artifact()
    _GTC = MediRelyGTC(topk=10)
    _RUNTIME["backend_label"] = TK.backend_label(_GTC.ms_backend)[0]
    _RUNTIME["is_cuvs"] = (_GTC.ms_backend == "cuvs")
    _print_checks("CLUSTER / H200", {
        "XRV model + classifier": _GTC.model is not None,
        "MedSigLIP encoder+bank": _GTC.ms_ready,
        "genuine cuVS (MedSigLIP)": _GTC.ms_backend == "cuvs",
        "reliability calibration": _GATE is not None,
        "demo cases": all(os.path.isfile(_SOURCES[g["key"]]["image_path"]) for g in GALLERY),
    }, extra=f"  backend = {_GTC.ms_backend!r}")


def startup_local_gpu():
    """Windows local NVIDIA GPU REAL mode. Honest device/backend reporting."""
    global _GTC, _GATE, _SOURCES, _IMAGE_ROOT, _RUNTIME
    import torch
    _patch_torch_load_compat()
    if not torch.cuda.is_available():
        print("\nNVIDIA CUDA GPU was not detected.\n"
              "Use --preview for CPU-only interface preview.\n")
        raise SystemExit(2)
    props = torch.cuda.get_device_properties(0)
    gpu = torch.cuda.get_device_name(0)
    vram_gb = props.total_memory / 1e9
    print("=" * 54, "\n MediRely GTC demo — LOCAL GPU device check")
    print(f"  GPU                : {gpu}")
    print(f"  total VRAM         : {vram_gb:.1f} GB")
    print(f"  PyTorch            : {torch.__version__}")
    print(f"  torch.version.cuda : {torch.version.cuda}")

    from gtc_demo.app.local_config import LocalConfig
    from gtc_demo.app.adapters import LogisticGate
    from gtc_demo.pipeline import MediRelyGTC
    cfg = LocalConfig()
    missing = cfg.missing_required()
    if missing:
        print("\nMissing required local artifacts/models — cannot run Local GPU mode:")
        for k, v in missing.items():
            print(f"  [MISSING] {k}: {v}")
        print("\nStage them from the H200 project (see WINDOWS_LOCAL_GPU.md).\n")
        raise SystemExit(3)

    raw = os.environ.get("MEDIRELY_RETRIEVAL_BACKEND", "auto").lower()
    prefer = {"auto": "cuvs", "cuvs": "cuvs", "torch": "torch"}.get(raw, "cuvs")
    strict_cuvs = (raw == "cuvs")   # explicit cuVS request → NEVER silently fall back
    ms_dtype = os.environ.get("MEDIRELY_MS_DTYPE", "float32")

    _GATE = LogisticGate.from_artifact(cfg.comparison())
    _SOURCES = json.load(open(cfg.gallery_sources(), encoding="utf-8"))
    _IMAGE_ROOT = cfg.image_root_or_none()
    try:
        _GTC = MediRelyGTC(config_path=cfg.config_yaml(), checkpoint=cfg.checkpoint(),
                           xrv_bank=cfg.xrv_bank(), ms_emb=cfg.ms_emb(),
                           ms_model_path=cfg.ms_model_path(), topk=10, device="cuda",
                           retrieval_prefer=prefer, ms_dtype=ms_dtype,
                           retrieval_allow_fallback=not strict_cuvs)
    except RuntimeError as exc:
        print(f"\nRequested retrieval backend 'cuvs' but genuine NVIDIA cuVS is "
              f"unavailable:\n  {exc}\n"
              "Install cuVS (WSL2 / Linux) or set MEDIRELY_RETRIEVAL_BACKEND=torch "
              "for the PyTorch CUDA fallback.\n")
        raise SystemExit(4)
    if strict_cuvs and _GTC.ms_backend != "cuvs":
        print(f"\nStrict cuVS requested but backend resolved to {_GTC.ms_backend!r}; "
              "aborting rather than mislabelling the backend.\n")
        raise SystemExit(4)

    _RUNTIME["gpu"] = gpu
    _RUNTIME["vram"] = f"{round(vram_gb)} GB VRAM"
    _RUNTIME["is_cuvs"] = (_GTC.ms_backend == "cuvs")
    _RUNTIME["backend_label"] = TK.backend_label(_GTC.ms_backend)[0]
    checks = {
        "RTX 2060 / CUDA GPU visible": torch.cuda.is_available(),
        "MedSigLIP encoder + bank": _GTC.ms_ready,
        "XRV model + classifier": _GTC.model is not None,
        "reliability calibration": _GATE is not None,
        "demo cases": all(os.path.isfile(_SOURCES[g["key"]]["image_path"]) for g in GALLERY),
    }
    if strict_cuvs:   # genuine cuVS was required — it built successfully or we aborted
        checks["genuine NVIDIA cuVS (import + index build)"] = (_GTC.ms_backend == "cuvs")
    _print_checks("LOCAL GPU", checks,
                  extra=f"  retrieval backend = {_GTC.ms_backend!r} → shown as "
                        f"'{_RUNTIME['backend_label']}'  |  image thumbnails: "
                        f"{'staged' if _IMAGE_ROOT else 'placeholder (bank_images not staged)'}")


def startup_preview():
    global _PREVIEW
    for g in GALLERY:
        p = os.path.join(ASSETS, f"{g['key']}.json")
        if not os.path.isfile(p):
            raise RuntimeError(f"missing preview asset: {p}")
        _PREVIEW[g["key"]] = json.load(open(p, encoding="utf-8"))
    _RUNTIME["is_cuvs"] = True
    _print_checks("LOCAL PREVIEW",
                  {f"demo asset {g['key']}": True for g in GALLERY},
                  extra=f"  recorded backend = "
                        f"{_PREVIEW[GALLERY[0]['key']].get('actual_backend','?')!r} (no live claim)")


# =========================== helpers ===================================
def _query_b64_for_example(key):
    return _PREVIEW[key].get("query_image_b64") if MODE == "preview" else \
        _file_b64(_SOURCES[key]["image_path"])


def _preview_adapt(gui):
    g = dict(gui)
    g["backend_chip"] = "Recorded • " + g.get("backend_label", "—")
    return g


def _footer(gui=None):
    return C.footer_html(gui, default_backend=_RUNTIME["backend_label"])


def _reset(state, b64, explain):
    return (state, C.query_panel_html(b64), explain, C.center_html(None),
            C.right_html(None), _footer(None), C.advanced_html(None))


# =========================== handlers ==================================
def on_select(key):
    return _reset({"kind": "ex", "key": key}, _query_b64_for_example(key), _explain_html(key))


def on_upload(path):
    if not path or not _valid_image(path):
        q = ('<div class="mr-panel"><div class="mr-eyebrow">PATIENT STUDY</div>'
             '<div class="mr-viewport"><div class="empty">Unreadable image<div class="sub">'
             'please upload a PNG / JPG / JPEG chest X-ray</div></div></div></div>')
        return (None, q, _explain_html(), C.center_html(None), C.right_html(None),
                _footer(None), C.advanced_html(None))
    return _reset({"kind": "up", "path": path}, _file_b64(path), _explain_html(upload=True))


def _run_real(path, label):
    from gtc_demo.app.adapters import to_gui_result
    t0 = time.perf_counter()
    pred = _GTC.predict(path, retrieval_encoder="medsiglip",
                        retrieval_backend="cuvs", gate=_GATE)
    dt = (time.perf_counter() - t0) * 1000.0
    pred["_query_b64"] = _file_b64(path)
    gui = to_gui_result(pred, label, dt, image_root=_IMAGE_ROOT)
    return C.center_html(gui), C.right_html(gui), _footer(gui), C.advanced_html(gui)


def on_recover(state):
    if not state:
        return C.center_html(None), C.right_html(None), _footer(None), C.advanced_html(None)
    try:
        if state["kind"] == "ex":
            key = state["key"]
            if MODE == "preview":
                gui = _preview_adapt(_PREVIEW[key])
                return (C.center_html(gui), C.right_html(gui), _footer(gui),
                        C.advanced_html(gui))
            return _run_real(_SOURCES[key]["image_path"], BYKEY[key]["label"])
        else:
            if MODE == "preview":
                return (C.preview_message_html(), C.right_html(None), _footer(None),
                        C.advanced_html(None))
            return _run_real(state["path"], "Uploaded Study")
    except Exception as exc:
        err = (f'<div class="mr-panel"><div class="mr-eyebrow">INFERENCE ERROR</div>'
               f'<div class="mr-note" style="color:var(--coral);margin-top:10px">'
               f'{type(exc).__name__}: {exc}</div></div>')
        return err, C.right_html(None), _footer(None), C.advanced_html(None)


# =========================== build =====================================
def build():
    choices = [(g["label"], g["key"]) for g in GALLERY]
    with gr.Blocks(css=CSS, title="MediRely", theme=gr.themes.Base()) as demo:
        state = gr.State(None)
        gr.HTML(C.header_html(mode=MODE, gpu=_RUNTIME["gpu"],
                              backend=_RUNTIME["backend_label"]))
        with gr.Row(elem_classes="mr-body", equal_height=True):
            with gr.Column(scale=27, min_width=380, elem_classes="col-left"):
                query = gr.HTML(C.query_panel_html(None))
                gr.HTML('<div class="mr-select-label">TRY A DEMO CASE</div>')
                selector = gr.Dropdown(choices=choices, value=None, show_label=False,
                                       container=False, filterable=False,
                                       elem_classes="mr-select")
                explain = gr.HTML(_explain_html())
                upload = gr.UploadButton("UPLOAD CHEST X-RAY", file_types=["image"],
                                         type="filepath", elem_classes="mr-upload-btn")
                recover = gr.Button("RECOVER WITH MEDIRELY", elem_classes="mr-primary")
                gr.HTML('<div class="mr-tagline">Retrieve, don\'t hallucinate.<br/>'
                        'Recover, but verify.</div>')
            with gr.Column(scale=49, elem_classes="col-center"):
                center = gr.HTML(C.center_html(None))
            with gr.Column(scale=24, min_width=320, elem_classes="col-right"):
                right = gr.HTML(C.right_html(None))
        footer = gr.HTML(_footer(None))
        with gr.Accordion("Deployment & Retrieval", open=False, elem_classes="mr-accordion"):
            gr.HTML(C.deployment_html(mode=MODE, runtime=_RUNTIME, parity=C.load_parity()))
        with gr.Accordion("Why should I trust this?", open=False, elem_classes="mr-accordion"):
            advanced = gr.HTML(C.advanced_html(None))

        sel_outs = [state, query, explain, center, right, footer, advanced]
        selector.change(on_select, inputs=selector, outputs=sel_outs)
        upload.upload(on_upload, inputs=upload, outputs=sel_outs)
        recover.click(on_recover, inputs=state, outputs=[center, right, footer, advanced])
    return demo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preview", action="store_true", help="CPU-only UI preview")
    ap.add_argument("--local-gpu", dest="local_gpu", action="store_true",
                    help="REAL inference on a local Windows NVIDIA GPU")
    args = ap.parse_args()
    global MODE
    MODE = "preview" if args.preview else ("local" if args.local_gpu else "cluster")

    {"preview": startup_preview, "local": startup_local_gpu,
     "cluster": startup_real}[MODE]()

    demo = build()
    port = int(os.environ.get("MR_PORT", "7860"))
    if MODE == "preview":
        print(f"[app] LOCAL PREVIEW → http://127.0.0.1:{port}")
        demo.launch(server_name="127.0.0.1", server_port=port, inbrowser=True,
                    show_api=False, quiet=True)
    elif MODE == "local":
        print(f"[app] LOCAL GPU → http://127.0.0.1:{port}")
        demo.launch(server_name="127.0.0.1", server_port=port, inbrowser=True,
                    show_api=False, quiet=True)
    else:
        print(f"[app] launching on 0.0.0.0:{port} (tunnel this port to view)")
        demo.launch(server_name="0.0.0.0", server_port=port, share=False,
                    show_api=False, quiet=True)


if __name__ == "__main__":
    main()
