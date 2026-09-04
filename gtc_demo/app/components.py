"""HTML builders for the MediRely GTC UI (premium dark navy workstation)."""
import json
import math
import os

from gtc_demo.app.tokens import (COL, RESEARCH_REFERENCE, DISCLAIMER, MICCAI_NOTE,
                                  MODEL_AUC)

_PARITY_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "artifacts", "cuvs_parity.json")


def load_parity():
    """Load the genuine cuVS-vs-PyTorch parity artifact. Returns dict or None;
    never fabricates values, never breaks startup."""
    try:
        with open(_PARITY_PATH, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

_GAUGE_C = 2 * math.pi * 52     # gauge circle circumference (r=52)


# ---------------------------------------------------------------- header
def header_html(mode="cluster", gpu=None, backend="NVIDIA cuVS"):
    if mode == "preview":
        pill = '<div class="mr-pill preview"><span class="dot"></span>LOCAL PREVIEW</div>'
    elif mode == "local":
        g = f" · {gpu}" if gpu else ""
        pill = f'<div class="mr-pill ready"><span class="dot"></span>LOCAL GPU{g}</div>'
    else:
        pill = '<div class="mr-pill ready"><span class="dot"></span>SYSTEM READY</div>'
    return f"""
<div class="mr-header">
  <div class="mr-brand">
    <div class="mr-mark"><span></span></div>
    <div>
      <div class="mr-title">MediRely</div>
      <div class="mr-sub">Reliable Clinical Context Recovery</div>
    </div>
  </div>
  <div class="mr-headright">
    <div class="mr-pill nv"><span class="dot"></span>{backend} · GPU Clinical Memory Search</div>
    {pill}
  </div>
</div>
<div class="mr-subbar">
  <div class="mr-def">Keeps multimodal medical AI working when clinical reports are missing.</div>
  <div class="mr-flow">
    <span class="fstep warn">REPORT MISSING</span><span class="farrow">→</span>
    <span class="fstep">REAL EVIDENCE</span><span class="farrow">→</span>
    <span class="fstep go">RECOVER + VERIFY</span>
  </div>
  <div class="mr-research">Research foundation · <b>MICCAI 2026 ML-CDS</b></div>
</div>"""


# ---------------------------------------------------------------- left
def query_panel_html(query_b64):
    if query_b64:
        vp = f'<img src="{query_b64}"/>'
        badge = '<div class="mr-badge avail">AVAILABLE</div>'
    else:
        vp = ('<div class="empty">Upload a chest X-ray'
              '<div class="sub">or try a demo case below</div></div>')
        badge = '<div class="mr-badge wait">AWAITING</div>'
    doc = ('<svg viewBox="0 0 24 24" width="20" height="20" fill="none" '
           'stroke="currentColor" stroke-width="1.7"><path d="M14 3H7a2 2 0 0 0-2 2'
           'v14a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V8z"/><path d="M14 3v5h5"/>'
           '<path d="M9.5 14.5l5 0" stroke-dasharray="1.4 1.6"/></svg>')
    return f"""
<div class="mr-panel">
  <div class="mr-eyebrow">PATIENT STUDY</div>
  <div class="mr-viewport"><span class="ct tl"></span><span class="ct tr"></span>
    <span class="ct bl"></span><span class="ct br"></span>{vp}</div>
  <div class="mr-inputrow"><span class="mr-io-label">CHEST X-RAY</span>{badge}</div>
  <div class="mr-inputrow"><span class="mr-io-label">CLINICAL REPORT</span>
    <div class="mr-badge missing">MISSING</div></div>
  <div class="mr-missing-card"><span class="ic">{doc}</span>
    <span>No clinical report provided. MediRely will recover relevant clinical
    context for this case.</span></div>
</div>"""


# --------------------------------------------------------------- center
def _card(c, idx):
    tagcls = "tag normal" if c.get("tag_normal") else "tag pos"
    return f"""
  <div class="mr-card" style="animation-delay:{idx*90}ms">
    <div class="rank">{c['rank']}</div>
    <div class="cvp"><img src="{c['image_b64']}"/></div>
    <div class="cbody">
      <div class="crow"><span class="sl">SIMILARITY</span>
        <span class="sv">{c['similarity']}%</span></div>
      <div class="rl">REAL REPORT</div>
      <div class="ex">{c['excerpt']}</div>
      <div class="{tagcls}">{c['tag']}</div>
    </div>
  </div>"""


def _consensus(rc):
    if not rc["consensus"]:
        return ('<div class="mr-chiprow"><span class="mr-warnchip">Neighbours '
                'disagree — no strong consensus</span></div>')
    chips = "".join(f'<span class="mr-echip">{name} · {cnt}/{rc["n_neighbors"]}</span>'
                    for name, cnt in rc["consensus"][:3])
    return f'<div class="mr-chiprow">{chips}</div>'


def center_html(gui):
    if gui is None:
        return """
<div class="mr-panel grow">
  <div class="mr-eyebrow">REAL RETRIEVED EVIDENCE</div>
  <div class="mr-ph">Clinical report is missing.<br/>Click
  <b style="color:var(--cyan)">Recover with MediRely</b> to retrieve real similar
  cases and estimate evidence reliability.</div>
</div>"""
    cards = "".join(_card(c, i) for i, c in enumerate(gui["evidence_cards"]))
    rc = gui["recovered_context"]
    return f"""
<div class="mr-panel">
  <div class="mr-evhead"><div class="mr-eyebrow">REAL RETRIEVED EVIDENCE</div>
    <div class="mr-searchtag">Retrieved with {gui['backend_label']}</div></div>
  <div class="mr-cards">{cards}</div>
</div>
<div class="mr-panel mr-context">
  <div class="mr-eyebrow">RECOVERED CLINICAL CONTEXT</div>
  <div class="mr-note">Consensus from real retrieved reports — no report is generated.</div>
  {_consensus(rc)}
  <div class="mr-note dim">{rc['agreeing']} neighbours agree on the consensus finding</div>
</div>"""


# ---------------------------------------------------------------- right
def _gauge(pct, color):
    off = _GAUGE_C * (1 - pct / 100.0)
    return f"""
  <svg class="gauge" viewBox="0 0 120 120">
    <circle class="gtrack" cx="60" cy="60" r="52"/>
    <circle class="gfill" cx="60" cy="60" r="52" stroke="{color}"
      stroke-dasharray="{_GAUGE_C:.1f}" stroke-dashoffset="{off:.1f}"
      transform="rotate(-90 60 60)"/>
    <text class="gval" x="60" y="57">{pct}%</text>
    <text class="glbl" x="60" y="76">EVIDENCE</text>
  </svg>"""


def right_html(gui):
    if gui is None:
        return ('<div class="mr-panel grow"><div class="mr-eyebrow">EVIDENCE '
                'RELIABILITY</div><div class="mr-ph small">Awaiting recovery.</div>'
                '</div>')
    c = COL[gui["reliability_color"]]
    io_, md = gui["image_only"], gui["medirely"]
    top = "".join(
        f'<div class="fr"><span class="fn">{f["name"]}</span>'
        f'<span class="fv"><span class="io">{f["image_only"]}%</span> '
        f'<span class="ar">→</span> <span class="md">{f["medirely"]}%</span></span></div>'
        for f in gui["full_findings"][:3])
    return f"""
<div class="mr-panel mr-rel">
  <div class="mr-eyebrow">EVIDENCE RELIABILITY</div>
  <div class="mr-gaugewrap">{_gauge(gui['evidence_reliability_pct'], c)}
    <div class="mr-lvl" style="color:{c}">{gui['reliability_level']}</div></div>
  <div class="mr-sub"><div class="r"><span>Neighbor Agreement</span>
    <b>{gui['neighbor_agreement_pct']}%</b></div>
    <div class="bar"><i style="width:{gui['neighbor_agreement_pct']}%"></i></div></div>
  <div class="mr-sub"><div class="r"><span>Retrieval Confidence</span>
    <b>{gui['retrieval_confidence_pct']}%</b></div>
    <div class="bar"><i style="width:{gui['retrieval_confidence_pct']}%"></i></div></div>
  <div class="mr-interp" style="border-color:{c}">{gui['reliability_text']}</div>
  <div class="mr-note dim">{gui['calibration_label']}</div>
</div>
<div class="mr-panel mr-pred">
  <div class="mr-eyebrow">PREDICTION · {io_['finding']}</div>
  <div class="mr-predrow">
    <div class="pc io"><span class="k">IMAGE ONLY</span><span class="p">{io_['pct']}%</span></div>
    <span class="ar">→</span>
    <div class="pc md"><span class="k">MEDIRELY</span><span class="p">{md['pct']}%</span></div>
  </div>
  <div class="mr-callout">{gui['prediction_callout']}</div>
  <div class="mr-toplist">{top}</div>
</div>"""


# ------------------------------------------------ preview upload message
def preview_message_html():
    return """
<div class="mr-panel grow">
  <div class="mr-eyebrow">REAL RETRIEVED EVIDENCE</div>
  <div class="mr-msg"><div class="t">Live recovery is available in Real Mode.</div>
    <div class="s">Preview Mode renders the interface locally without loading
    MedSigLIP or NVIDIA cuVS. Try a demo case to see real recovered evidence, or run
    the app in Real Mode on a GPU host for live recovery of your upload.</div></div>
</div>"""


# ---------------------------------------------------------------- footer
def _status(is_current, mode):
    if mode == "preview":
        return "PREVIEW", "amber"
    if is_current:
        return "LIVE", "green"
    return "VALIDATED", "cyan"


def deployment_html(mode="cluster", runtime=None, parity=None):
    """Deployment & Retrieval section. Runtime-honest: the LOCAL card is LIVE only
    when the current runtime is the local GPU; the cuVS card is LIVE only on the
    H200/cluster runtime with genuine cuVS. Benchmark values come from the parity
    JSON artifact (or an 'unavailable' note) — never fabricated."""
    runtime = runtime or {}
    is_cuvs = runtime.get("is_cuvs")
    backend_label = runtime.get("backend_label", "NVIDIA cuVS")
    # LOCAL card tells the consumer-GPU story; use the live GPU/backend only in local mode
    if mode == "local":
        gpu = runtime.get("gpu") or "NVIDIA GeForce RTX 2060"
        vram = runtime.get("vram") or "6 GB VRAM"
        local_backend = backend_label + (" · GPU brute-force" if is_cuvs
                                         else " · Exact vector search")
    else:
        gpu, vram = "NVIDIA GeForce RTX 2060", "6 GB VRAM"
        local_backend = "PyTorch CUDA · Exact vector search"
    local_live = (mode == "local")
    ls, lc = _status(local_live, mode)
    # cuVS card is LIVE whenever cuVS is the current runtime backend (local OR cluster).
    cuvs_live = is_cuvs and mode in ("local", "cluster")
    cs, cc = _status(cuvs_live, mode)
    if cuvs_live:
        # cuVS running live here — do NOT frame it as "validated on H200".
        cuvs_sub = "Genuine GPU brute-force retrieval"
        cuvs_desc = ("NVIDIA cuVS runs live as the GPU clinical-memory retrieval "
                     "backend and scales to larger clinical memory banks.")
    else:
        cuvs_sub = "Validated on NVIDIA H200 research infrastructure"
        cuvs_desc = ("cuVS provides the scalable GPU retrieval backend for larger "
                     "clinical memory banks.")
        if mode == "preview":
            cuvs_desc = "Recorded • NVIDIA cuVS. " + cuvs_desc

    if parity:
        pj = parity
        bench = f"""
    <div class="mr-bench">
      <div class="bh">{pj.get('num_queries','—')} held-out queries · k={pj.get('k','—')}</div>
      <div class="br"><span>Top-1 agreement</span><b>{100*pj.get('top1_agreement',0):.1f}%</b></div>
      <div class="br"><span>Top-K set agreement</span><b>{100*pj.get('topk_set_agreement',0):.1f}%</b></div>
      <div class="br"><span>Mean Jaccard</span><b>{100*pj.get('mean_jaccard',0):.1f}%</b></div>
      <div class="bf">{pj.get('notes','')}</div>
    </div>"""
    else:
        bench = '<div class="mr-bench"><div class="bf">Validation artifact unavailable.</div></div>'

    return f"""
<div class="mr-depcards">
  <div class="mr-depcard">
    <div class="dh">LOCAL DEPLOYMENT</div>
    <div class="dg">{gpu}</div><div class="ds">{vram}</div>
    <div class="dk">Backend</div><div class="dv">{local_backend}</div>
    <div class="mr-dstatus {lc}">{ls}</div>
    <div class="dd">End-to-end MediRely inference runs locally on a consumer NVIDIA GPU.</div>
  </div>
  <div class="mr-depcard">
    <div class="dh">SCALABLE RETRIEVAL</div>
    <div class="dg">NVIDIA cuVS</div><div class="ds">{cuvs_sub}</div>
    <div class="mr-dstatus {cc}">{cs}</div>
    <div class="dd">{cuvs_desc}</div>
    {bench}
  </div>
</div>"""


def footer_html(gui=None, default_backend="NVIDIA cuVS"):
    backend = (gui.get("backend_chip") or gui["backend_label"]) if gui else default_backend
    return f"""
<div class="mr-footer">
  <div class="fl"><span class="fk">RETRIEVAL ENGINE</span><span class="fv">{backend}</span>
    <span class="sep"></span><span class="fk">EMBEDDING</span><span class="fv">MedSigLIP</span></div>
  <div class="fc"><span class="fk">RESEARCH FOUNDATION</span>
    <span class="fv">Accepted at MICCAI 2026 ML-CDS</span></div>
  <div class="fr">{DISCLAIMER}</div>
</div>"""


# -------------------------------------------------------------- advanced
def _architecture_html():
    return """
<div class="mr-arch">
  <div class="arow"><span class="al">NORMAL MULTIMODAL AI</span>
    <span class="af">X-ray + clinical report&nbsp; → &nbsp;Multimodal AI&nbsp; → &nbsp;Prediction</span></div>
  <div class="arow"><span class="al warn">WHEN THE REPORT IS MISSING</span>
    <span class="af">X-ray + <span class="miss">report missing</span>&nbsp; → &nbsp;degraded prediction</span></div>
  <div class="arow"><span class="al go">WITH MEDIRELY</span>
    <span class="af">X-ray&nbsp; → &nbsp;<b>retrieve real cases → recover context → estimate reliability</b>
    &nbsp; → &nbsp;Multimodal AI&nbsp; → &nbsp;Prediction</span></div>
</div>"""


def advanced_html(gui):
    grid = ""
    if gui is not None:
        grid = f"""
<div class="mr-adv">
  <div><div class="k">EVIDENCE ENCODER</div><div class="v">MedSigLIP</div></div>
  <div><div class="k">SEARCH ENGINE</div><div class="v">{gui['backend_label']}</div></div>
  <div><div class="k">EVIDENCE SOURCE</div><div class="v">Real training cases (Top-K)</div></div>
  <div><div class="k">NEIGHBOR AGREEMENT</div><div class="v">{gui['neighbor_agreement_pct']}%</div></div>
  <div><div class="k">RETRIEVAL CONFIDENCE</div><div class="v">{gui['retrieval_confidence_pct']}%</div></div>
  <div><div class="k">GTC EVIDENCE RELIABILITY</div><div class="v">{gui['evidence_reliability_pct']}% · {gui['reliability_level']}</div></div>
  <div><div class="k">CALIBRATION</div><div class="v">Validation-calibrated (MedSigLIP)</div></div>
  <div><div class="k">CLASSIFIER INPUT</div><div class="v">Validated XRV representation</div></div>
  <div><div class="k">INFERENCE</div><div class="v">{gui['total_latency_ms']} ms</div></div>
</div>"""
    return f"""
{_architecture_html()}
<div class="mr-advmetric"><span class="k">{MODEL_AUC['source']}</span>
  <span>Image-only <b>{MODEL_AUC['image_only']:.3f}</b></span>
  <span class="ar">→</span>
  <span class="md">MediRely <b>{MODEL_AUC['medirely']:.3f}</b></span>
  <span class="gtc">GTC extension</span></div>
{grid}
<div class="mr-note" style="margin-top:12px">The original MediRely research established the
missing-context recovery framework. This GTC demo extends it with an interactive application,
MedSigLIP retrieval, and NVIDIA cuVS integration.</div>"""
