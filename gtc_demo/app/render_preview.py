"""Faithful 1920x1080 Pillow renderer of the redesigned MediRely workstation.

The compute container has no browser/fonts, so we render the design directly with
Pillow (DejaVu fonts from matplotlib) using REAL inference data. Approximates the
Gradio 3-column layout for screenshot review; live validation is via the browser
(preview mode). Consumes the same gui dict as the app components.
"""
from __future__ import annotations

import base64
import io

from PIL import Image, ImageDraw, ImageFont
from matplotlib import font_manager

from gtc_demo.app.tokens import COL, MODEL_AUC

W, H = 1920, 1080
_SANS = font_manager.findfont("DejaVu Sans")
_BOLD = font_manager.findfont(font_manager.FontProperties(family="DejaVu Sans", weight="bold"))


def _f(sz, bold=False):
    return ImageFont.truetype(_BOLD if bold else _SANS, sz)


def _hx(c):
    c = c.lstrip("#")
    return tuple(int(c[i:i + 2], 16) for i in (0, 2, 4))


def _rr(d, box, r, fill=None, outline=None, width=1):
    d.rounded_rectangle(box, radius=r, fill=fill, outline=outline, width=width)


def _t(d, xy, s, font, fill, anchor="la", track=0):
    if track == 0:
        d.text(xy, s, font=font, fill=fill, anchor=anchor)
        return
    total = sum(d.textlength(ch, font=font) for ch in s) + track * max(0, len(s) - 1)
    x, y = xy
    if anchor[0] == "r":
        x -= total
    elif anchor[0] == "m":
        x -= total / 2
    for ch in s:
        d.text((x, y), ch, font=font, fill=fill, anchor="la")
        x += d.textlength(ch, font=font) + track


def _datauri(uri):
    return Image.open(io.BytesIO(base64.b64decode(uri.split(",", 1)[1]))).convert("RGB")


def _place(base, img, box):
    x0, y0, x1, y1 = box
    bw, bh = x1 - x0, y1 - y0
    s = min((bw - 12) / img.width, (bh - 12) / img.height)
    fit = img.resize((max(1, int(img.width * s)), max(1, int(img.height * s))), Image.LANCZOS)
    base.paste(fit, (x0 + (bw - fit.width) // 2, y0 + (bh - fit.height) // 2))


def _panel(d, box, r=14):
    _rr(d, box, r, fill=_hx(COL["card"]), outline=_hx(COL["border"]), width=1)


def _bar(d, box, frac, color, track="#22344d"):
    x0, y0, x1, y1 = box
    _rr(d, box, (y1 - y0) // 2, fill=_hx(track))
    _rr(d, (x0, y0, x0 + int((x1 - x0) * max(0.02, min(1, frac))), y1), (y1 - y0) // 2, fill=_hx(color))


def _wrap(d, text, font, fill, xy, maxw, lh, lines):
    x, y = xy
    words = text.split()
    line = ""
    n = 0
    for w in words:
        t = (line + " " + w).strip()
        if d.textlength(t, font=font) <= maxw:
            line = t
        else:
            d.text((x, y), line, font=font, fill=fill)
            y += lh
            n += 1
            line = w
            if n >= lines - 1:
                break
    if n < lines and line:
        d.text((x, y), line, font=font, fill=fill)


def render(gui, out_path):
    im = Image.new("RGB", (W, H), _hx(COL["bg"]))
    d = ImageDraw.Draw(im)
    txt, dim, muted, faint = (_hx(COL["text"]), _hx(COL["dim"]), _hx(COL["muted"]),
                              _hx(COL["faint"]))
    cyan, green, purple = _hx(COL["cyan"]), _hx(COL["green"]), _hx(COL["purple"])
    lvl = COL[gui["reliability_color"]]
    lvlc = _hx(lvl)

    # ---------- header ----------
    _rr(d, (30, 28, 74, 72), 12, fill=_hx("#0a2536"), outline=_hx("#1f4d68"), width=1)
    d.ellipse((44, 42, 60, 58), outline=cyan, width=3)
    d.ellipse((49, 47, 55, 53), fill=green)
    _t(d, (88, 22), "MediRely", _f(34, bold=True), txt)
    _t(d, (90, 66), "Reliable Clinical Context Recovery", _f(15), dim)
    # right pills
    def pill(x1, label, col, dot):
        w = int(d.textlength(label, _f(13, bold=True))) + 46
        box = (x1 - w, 30, x1, 62)
        _rr(d, box, 16, fill=_hx(COL["panel"]), outline=_hx(COL["border2"]), width=1)
        d.ellipse((box[0] + 15, 42, box[0] + 24, 51), fill=dot)
        _t(d, (box[0] + 32, 38), label, _f(13, bold=True), col)
        return box[0]
    x = pill(W - 30, "SYSTEM READY", txt, green)
    backend_lbl = gui.get("backend_label", "NVIDIA cuVS")
    pill(x - 12, f"{backend_lbl} · GPU Clinical Memory Search", cyan, cyan)

    # ---------- sub-bar: definition · concept flow · research ----------
    _t(d, (30, 92), "Keeps multimodal medical AI working when clinical reports are missing.",
       _f(13), dim)
    def chip(x0, label, fg, bg, bd):
        w = int(d.textlength(label, _f(11, bold=True))) + 24
        _rr(d, (x0, 88, x0 + w, 112), 8, fill=_hx(bg), outline=_hx(bd), width=1)
        _t(d, (x0 + 12, 93), label, _f(11, bold=True), fg, track=.5)
        return x0 + w
    fx = W // 2 - 205
    fx = chip(fx, "REPORT MISSING", _hx(COL["coral"]), "#1a1014", "#3a1f27")
    _t(d, (fx + 7, 91), "→", _f(15, bold=True), muted); fx += 28
    fx = chip(fx, "REAL EVIDENCE", dim, COL["chip"], COL["border2"])
    _t(d, (fx + 7, 91), "→", _f(15, bold=True), muted); fx += 28
    chip(fx, "RECOVER + VERIFY", cyan, "#08202f", "#1e4a63")
    _t(d, (W - 30, 94), "Research foundation · MICCAI 2026 ML-CDS", _f(12.5), muted, anchor="ra")
    d.line((30, 122, W - 30, 122), fill=_hx(COL["border"]), width=1)

    top, bot = 140, 986
    # ================= LEFT: PATIENT STUDY =================
    lx0, lx1 = 30, 512
    _panel(d, (lx0, top, lx1, bot))
    _t(d, (lx0 + 18, top + 16), "PATIENT STUDY", _f(13, bold=True), dim, track=1.5)
    vp = (lx0 + 18, top + 44, lx1 - 18, top + 372)
    _rr(d, vp, 10, fill=_hx(COL["viewport"]))
    if gui.get("query_image_b64"):
        _place(im, _datauri(gui["query_image_b64"]), vp)
        d = ImageDraw.Draw(im)
    # input rows
    iy = top + 392
    _t(d, (lx0 + 18, iy), "CHEST X-RAY", _f(14, bold=True), txt)
    _rr(d, (lx1 - 122, iy - 3, lx1 - 18, iy + 21), 6, fill=_hx("#123026"))
    _t(d, (lx1 - 70, iy), "AVAILABLE", _f(12, bold=True), green, anchor="ma", track=1)
    iy2 = iy + 36
    _t(d, (lx0 + 18, iy2), "CLINICAL REPORT", _f(14, bold=True), txt)
    _rr(d, (lx1 - 104, iy2 - 3, lx1 - 18, iy2 + 21), 6, fill=_hx("#331a22"))
    _t(d, (lx1 - 61, iy2 - 1), "MISSING", _f(13, bold=True), _hx(COL["coral"]), anchor="ma", track=1.5)
    # missing card
    mc = (lx0 + 18, iy2 + 34, lx1 - 18, iy2 + 96)
    _rr(d, mc, 10, fill=_hx("#170f13"), outline=_hx("#3a1f27"), width=1)
    _wrap(d, "No clinical report provided. MediRely will recover relevant clinical "
          "context for this case.", _f(12), dim, (mc[0] + 12, mc[1] + 11), mc[2] - mc[0] - 24, 17, 3)
    # demo case + explanation + buttons
    sy = iy2 + 112
    _t(d, (lx0 + 18, sy), "TRY A DEMO CASE", _f(12, bold=True), muted, track=1.6)
    _rr(d, (lx0 + 18, sy + 20, lx1 - 18, sy + 52), 9, fill=_hx(COL["card"]),
        outline=_hx(COL["border2"]), width=1)
    _t(d, (lx0 + 32, sy + 28), gui.get("case_label", "Strong Recovery"), _f(14, bold=True), txt)
    _t(d, (lx1 - 34, sy + 28), "▾", _f(14), muted)
    _wrap(d, gui.get("_explanation", ""), _f(12.5), dim, (lx0 + 24, sy + 66),
          lx1 - lx0 - 42, 17, 2)
    _rr(d, (lx0 + 18, bot - 96, lx1 - 18, bot - 62), 9, fill=_hx(COL["card"]),
        outline=_hx(COL["border2"]), width=1)
    _t(d, ((lx0 + lx1) // 2, bot - 86), "UPLOAD CHEST X-RAY", _f(13, bold=True), txt, anchor="ma", track=1)
    _rr(d, (lx0 + 18, bot - 52, lx1 - 18, bot - 14), 11, fill=cyan)
    _t(d, ((lx0 + lx1) // 2, bot - 42), "RECOVER WITH MEDIRELY", _f(16, bold=True), _hx("#03121c"), anchor="ma", track=1)

    # ================= CENTER: EVIDENCE + CONTEXT =================
    cx0, cx1 = 532, 1322
    ev = (cx0, top, cx1, top + 512)
    _panel(d, ev)
    _t(d, (cx0 + 18, top + 16), "REAL RETRIEVED EVIDENCE", _f(13, bold=True), dim, track=1.4)
    tag = "Retrieved with NVIDIA cuVS"
    tw = int(d.textlength(tag, _f(11.5, bold=True))) + 22
    _rr(d, (cx1 - 18 - tw, top + 12, cx1 - 18, top + 34), 12, fill=_hx("#0e2a3d"), outline=_hx("#1e4a63"), width=1)
    _t(d, (cx1 - 18 - tw + 11, top + 16), tag, _f(11.5, bold=True), cyan)
    cards = gui.get("evidence_cards", [])
    cy = top + 46
    for i, c in enumerate(cards[:3]):
        ch = 150
        cbox = (cx0 + 16, cy, cx1 - 16, cy + ch)
        _rr(d, cbox, 12, fill=_hx(COL["card"]), outline=_hx(COL["border2"]), width=1)
        # rank
        d.ellipse((cbox[0] + 14, cy + 14, cbox[0] + 40, cy + 40), outline=cyan, width=2, fill=_hx("#0e2436"))
        _t(d, (cbox[0] + 27, cy + 18), str(c.get("rank", i + 1)), _f(14, bold=True), cyan, anchor="ma")
        # thumb
        tb = (cbox[0] + 52, cy + 14, cbox[0] + 52 + 132, cy + ch - 14)
        _rr(d, tb, 8, fill=_hx(COL["viewport"]))
        _place(im, _datauri(c["image_b64"]), tb)
        d = ImageDraw.Draw(im)
        bx = cbox[0] + 200
        _t(d, (bx, cy + 16), "SIMILARITY", _f(11, bold=True), muted, track=1.3)
        _t(d, (cbox[2] - 16, cy + 12), f"{c['similarity']}%", _f(22, bold=True), cyan, anchor="ra")
        _t(d, (bx, cy + 42), "REAL REPORT", _f(10.5, bold=True), muted, track=1.3)
        _wrap(d, c["excerpt"], _f(12.5), dim, (bx, cy + 58), cbox[2] - bx - 16, 17, 2)
        tagn = c.get("tag", "No Finding")
        tcol = green if c.get("tag_normal") else _hx(COL["amber"])
        tbg = "#123026" if c.get("tag_normal") else "#2e2713"
        twd = int(d.textlength(tagn, _f(11, bold=True))) + 20
        _rr(d, (bx, cy + ch - 34, bx + twd, cy + ch - 14), 11, fill=_hx(tbg))
        _t(d, (bx + 10, cy + ch - 32), tagn, _f(11, bold=True), tcol)
        cy += ch + 12

    ctx = (cx0, top + 528, cx1, bot)
    _panel(d, ctx)
    _t(d, (cx0 + 18, ctx[1] + 14), "RECOVERED CLINICAL CONTEXT", _f(13, bold=True), dim, track=1.2)
    _t(d, (cx0 + 18, ctx[1] + 40), "Consensus from real retrieved reports — no report is generated.",
       _f(12), muted)
    rc = gui.get("recovered_context", {})
    chx = cx0 + 18
    chy = ctx[1] + 68
    if rc.get("consensus"):
        for name, cnt in rc["consensus"][:3]:
            lab = f"{name} · {cnt}/{rc.get('n_neighbors', 0)}"
            wd = int(d.textlength(lab, _f(13, bold=True))) + 24
            _rr(d, (chx, chy, chx + wd, chy + 32), 8, fill=_hx(COL["chip"]), outline=_hx(COL["border2"]), width=1)
            _t(d, (chx + 12, chy + 7), lab, _f(13, bold=True), txt)
            chx += wd + 10
    else:
        _t(d, (chx, chy + 4), "Neighbours disagree — no strong consensus", _f(13, bold=True), _hx(COL["amber"]))
    _t(d, (cx0 + 18, ctx[3] - 30), f"{rc.get('agreeing','—')} neighbours agree on the consensus finding",
       _f(11.5), faint)

    # ================= RIGHT: RELIABILITY + PREDICTION =================
    rx0, rx1 = 1342, 1890
    rel = (rx0, top, rx1, top + 470)
    _panel(d, rel)
    _t(d, (rx0 + 18, top + 14), "EVIDENCE RELIABILITY", _f(13, bold=True), dim, track=1.2)
    # gauge
    gcx, gcy, gr = (rx0 + rx1) // 2, top + 128, 66
    d.arc((gcx - gr, gcy - gr, gcx + gr, gcy + gr), 0, 360, fill=_hx("#22344d"), width=12)
    pct = gui["evidence_reliability_pct"]
    d.arc((gcx - gr, gcy - gr, gcx + gr, gcy + gr), -90, -90 + 3.6 * pct, fill=lvlc, width=12)
    _t(d, (gcx, gcy - 20), f"{pct}%", _f(32, bold=True), txt, anchor="ma")
    _t(d, (gcx, gcy + 16), "EVIDENCE", _f(9, bold=True), muted, anchor="ma", track=2)
    _t(d, (gcx, gcy + gr + 8), f"{gui['reliability_level']}", _f(16, bold=True), lvlc, anchor="ma", track=1.5)
    # sub bars
    sby = top + 250
    for label, key in (("Neighbor Agreement", "neighbor_agreement_pct"),
                       ("Retrieval Confidence", "retrieval_confidence_pct")):
        v = gui[key]
        _t(d, (rx0 + 18, sby), label, _f(13), dim)
        _t(d, (rx1 - 18, sby - 2), f"{v}%", _f(16, bold=True), txt, anchor="ra")
        _bar(d, (rx0 + 18, sby + 22, rx1 - 18, sby + 30), v / 100, COL["cyan"])
        sby += 46
    ib = (rx0 + 18, sby + 4, rx1 - 18, sby + 58)
    _rr(d, ib, 6, fill=_hx("#0c1524"))
    d.rectangle((ib[0], ib[1], ib[0] + 3, ib[3]), fill=lvlc)
    _wrap(d, gui.get("reliability_text", ""), _f(12), dim, (ib[0] + 12, ib[1] + 9), ib[2] - ib[0] - 22, 16, 3)
    _t(d, (rx0 + 18, rel[3] - 22), gui.get("calibration_label", ""), _f(11), faint)

    pr = (rx0, top + 486, rx1, bot)
    _panel(d, pr)
    io_, md = gui["image_only"], gui["medirely"]
    _t(d, (rx0 + 18, pr[1] + 14), f"PREDICTION · {io_['finding']}", _f(13, bold=True), dim, track=1)
    py = pr[1] + 46
    # image-only column
    _rr(d, (rx0 + 18, py, (rx0 + rx1) // 2 - 20, py + 80), 10, fill=_hx("#08243a"), outline=_hx("#1b4a68"), width=1)
    _t(d, ((rx0 + (rx0 + rx1) // 2 - 2) // 2 + 9, py + 12), "IMAGE ONLY", _f(11.5, bold=True), cyan, anchor="ma", track=1)
    _t(d, ((rx0 + (rx0 + rx1) // 2 - 2) // 2 + 9, py + 30), f"{io_['pct']}%", _f(32, bold=True), cyan, anchor="ma")
    _t(d, ((rx0 + rx1) // 2, py + 30), "→", _f(24, bold=True), muted, anchor="ma")
    # medirely column
    mx0 = (rx0 + rx1) // 2 + 20
    _rr(d, (mx0, py, rx1 - 18, py + 80), 10, fill=_hx("#1a1533"), outline=_hx("#3d306a"), width=1)
    _t(d, ((mx0 + rx1 - 18) // 2, py + 12), "MEDIRELY", _f(11.5, bold=True), purple, anchor="ma", track=1)
    _t(d, ((mx0 + rx1 - 18) // 2, py + 30), f"{md['pct']}%", _f(32, bold=True), purple, anchor="ma")
    _wrap(d, gui.get("prediction_callout", ""), _f(12.5), dim, (rx0 + 18, py + 96), rx1 - rx0 - 36, 17, 2)
    # top-list
    ty = py + 138
    d.line((rx0 + 18, ty - 8, rx1 - 18, ty - 8), fill=_hx(COL["border"]), width=1)
    for f in gui.get("full_findings", [])[:3]:
        _t(d, (rx0 + 18, ty), f["name"], _f(12.5), dim)
        _t(d, (rx1 - 120, ty), f"{f['image_only']}%", _f(12.5), cyan, anchor="ra")
        _t(d, (rx1 - 100, ty), "→", _f(12.5), faint)
        _t(d, (rx1 - 18, ty), f"{f['medirely']}%", _f(12.5, bold=True), purple, anchor="ra")
        ty += 24

    # ================= FOOTER =================
    fy = 998
    d.line((30, fy - 8, W - 30, fy - 8), fill=_hx(COL["border"]), width=1)
    _t(d, (30, fy), "RETRIEVAL ENGINE", _f(11, bold=True), muted, track=1)
    _t(d, (162, fy - 1), gui.get("backend_label", "NVIDIA cuVS"), _f(13, bold=True), txt)
    _t(d, (300, fy), "EMBEDDING", _f(11, bold=True), muted, track=1)
    _t(d, (388, fy - 1), "MedSigLIP", _f(13, bold=True), txt)
    # research foundation (quiet, centered) — MICCAI as credibility, not hero
    _t(d, (W // 2, fy), "RESEARCH FOUNDATION", _f(11, bold=True), muted, anchor="ma", track=1)
    _t(d, (W // 2, fy + 15), "Accepted at MICCAI 2026 ML-CDS", _f(12.5), dim, anchor="ma")
    _t(d, (W - 30, fy), "Research prototype • Not for clinical use", _f(11.5), faint, anchor="ra")

    im.save(out_path)
    return out_path
