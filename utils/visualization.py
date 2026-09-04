"""Matplotlib helpers shared by the figure-generation scripts.

These functions only produce *conceptual* diagrams (no experimental numbers).
Performance plots live in scripts/ and read real CSVs.
"""
import matplotlib

matplotlib.use("Agg")  # headless / SLURM safe
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch  # noqa: E402


# A muted, print-friendly palette.
PALETTE = {
    "image": "#4C72B0",
    "text": "#55A868",
    "bank": "#C44E52",
    "fusion": "#8172B3",
    "fallback": "#CCB974",
    "bg": "#F2F2F2",
    "edge": "#333333",
}


def box(ax, x, y, w, h, text, color, fontsize=9, text_color="white"):
    """Draw a rounded box centred at (x, y) with wrapped text."""
    patch = FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.05",
        linewidth=1.2, edgecolor=PALETTE["edge"], facecolor=color, alpha=0.95,
    )
    ax.add_patch(patch)
    ax.text(x, y, text, ha="center", va="center", fontsize=fontsize,
            color=text_color, wrap=True, zorder=5)
    return patch


def arrow(ax, x1, y1, x2, y2, style="-|>", color=None, lw=1.5, ls="-"):
    color = color or PALETTE["edge"]
    a = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style,
                        mutation_scale=14, linewidth=lw, color=color,
                        linestyle=ls, zorder=1)
    ax.add_patch(a)
    return a


def new_canvas(figsize=(8, 4.5)):
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")
    return fig, ax


def save(fig, path, also_png=True):
    import os
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, bbox_inches="tight", dpi=200)
    if also_png and path.endswith(".pdf"):
        fig.savefig(path[:-4] + ".png", bbox_inches="tight", dpi=200)
    plt.close(fig)
