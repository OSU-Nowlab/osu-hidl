"""Generate the OSU HPC-AI stack architecture diagram as a PNG."""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch

fig, ax = plt.subplots(figsize=(9, 7.5))
ax.set_xlim(0, 9)
ax.set_ylim(0, 7.5)
ax.axis("off")

# ── colour palette ────────────────────────────────────────────────────────────
GREEN_BG   = "#d9e8d0"   # Distributed DL Frameworks layer
BLUE_BG    = "#cfd9ea"   # Deep Learning Frameworks layer
PINK_BG    = "#f0d0d0"   # Communication Libraries layer
TAN_BG     = "#ede8d8"   # HPC Hardware outer
DARK_BG    = "#1a1a1a"   # HPC Hardware inner boxes
ITEM_PINK  = "#e8a0a0"   # component boxes (pink)
BORDER     = "#555555"

def rounded_box(ax, x, y, w, h, color, label=None, label_y_offset=None,
                bold_label=False, radius=0.25, lw=1.2, zorder=1):
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0,rounding_size={radius}",
        facecolor=color, edgecolor=BORDER, linewidth=lw, zorder=zorder,
    )
    ax.add_patch(box)
    if label:
        ly = y + h - 0.32 if label_y_offset is None else y + label_y_offset
        weight = "bold" if bold_label else "normal"
        ax.text(x + w / 2, ly, label, ha="center", va="center",
                fontsize=10, fontweight=weight, zorder=zorder + 1)

def component_box(ax, x, y, w, h, label, fontsize=9, zorder=5):
    """Pink component box with text."""
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0,rounding_size=0.15",
        facecolor=ITEM_PINK, edgecolor=BORDER, linewidth=1.0, zorder=zorder,
    )
    ax.add_patch(box)
    ax.text(x + w / 2, y + h / 2, label, ha="center", va="center",
            fontsize=fontsize, zorder=zorder + 1)

def dark_box(ax, x, y, w, h, label, fontsize=9, zorder=5):
    """Dark hardware box with white text."""
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0,rounding_size=0.12",
        facecolor=DARK_BG, edgecolor="#888888", linewidth=0.8, zorder=zorder,
    )
    ax.add_patch(box)
    ax.text(x + w / 2, y + h / 2, label, ha="center", va="center",
            fontsize=fontsize, color="white", zorder=zorder + 1)

# ── Layer dimensions ───────────────────────────────────────────────────────────
PAD_X = 0.35   # outer left/right padding
W_TOTAL = 9 - 2 * PAD_X   # 8.3

# ── Layer 1: Distributed DL Frameworks (top, green) ───────────────────────────
L1_Y, L1_H = 6.10, 1.10
rounded_box(ax, PAD_X, L1_Y, W_TOTAL, L1_H, GREEN_BG, zorder=2)
ax.text(PAD_X + W_TOTAL / 2, L1_Y + L1_H - 0.28,
        "Distributed DL Frameworks", ha="center", va="center",
        fontsize=10, fontweight="normal", zorder=3)

# Component boxes: DeepSpeed | vLLM | SGLang | PyTorch
comp_w, comp_h = 1.7, 0.52
comp_y = L1_Y + 0.14
gap = (W_TOTAL - 4 * comp_w) / 5
for i, name in enumerate(["DeepSpeed", "vLLM", "SGLang", "PyTorch"]):
    cx = PAD_X + gap + i * (comp_w + gap)
    component_box(ax, cx, comp_y, comp_w, comp_h, name, zorder=4)

# ── Arrows between layers ──────────────────────────────────────────────────────
arrow_kw = dict(arrowstyle="<->", color="#333333", lw=1.5)

def add_arrow(ax, y_start, y_end):
    ax.annotate("", xy=(4.5, y_end), xytext=(4.5, y_start),
                arrowprops=dict(arrowstyle="<->", color="#333333", lw=1.5),
                zorder=10)

# ── Layer 2: Deep Learning Frameworks (blue) ──────────────────────────────────
L2_Y, L2_H = 4.65, 1.10
add_arrow(ax, L1_Y, L2_Y + L2_H)
rounded_box(ax, PAD_X, L2_Y, W_TOTAL, L2_H, BLUE_BG, zorder=2)
ax.text(PAD_X + W_TOTAL / 2, L2_Y + L2_H - 0.28,
        "Deep Learning Frameworks", ha="center", va="center",
        fontsize=10, zorder=3)
pt_w, pt_h = 2.2, 0.52
component_box(ax, PAD_X + (W_TOTAL - pt_w) / 2, L2_Y + 0.14, pt_w, pt_h,
              "PyTorch", zorder=4)

# ── Layer 3: Communication Libraries (pink) ───────────────────────────────────
L3_Y, L3_H = 3.20, 1.10
add_arrow(ax, L2_Y, L3_Y + L3_H)
rounded_box(ax, PAD_X, L3_Y, W_TOTAL, L3_H, PINK_BG, zorder=2)
ax.text(PAD_X + W_TOTAL / 2, L3_Y + L3_H - 0.28,
        "Communication Libraries", ha="center", va="center",
        fontsize=10, zorder=3)
cl_w, cl_h = 2.4, 0.52
cl_y = L3_Y + 0.14
cx = PAD_X + (W_TOTAL - cl_w) / 2
component_box(ax, cx, cl_y, cl_w, cl_h, "MVAPICH-Plus", zorder=4)

# ── Layer 4: HPC Hardware (tan outer + two dashed sub-sections) ──────────────
L4_Y, L4_H = 0.20, 2.65
add_arrow(ax, L3_Y, L4_Y + L4_H)
rounded_box(ax, PAD_X, L4_Y, W_TOTAL, L4_H, TAN_BG, zorder=2)

# Sub-section: Compute (dashed border)
comp_sec_x, comp_sec_y = PAD_X + 0.15, L4_Y + 0.15
comp_sec_w, comp_sec_h = 3.30, 2.10
compute_box = FancyBboxPatch(
    (comp_sec_x, comp_sec_y), comp_sec_w, comp_sec_h,
    boxstyle="round,pad=0,rounding_size=0.2",
    facecolor=TAN_BG, edgecolor=BORDER, linewidth=1.2,
    linestyle="--", zorder=3,
)
ax.add_patch(compute_box)
# "Compute" label outside (above) the dashed box, centered
ax.text(comp_sec_x + comp_sec_w / 2, L4_Y + L4_H - 0.30,
        "Compute", ha="center", va="center",
        fontsize=10, fontweight="bold", zorder=4)

dk_w = comp_sec_w - 0.30
dk_h = 0.44
dk_x = comp_sec_x + 0.15
dk_gap = 0.24
total_boxes_h = 2 * dk_h + dk_gap
dk_start_y = comp_sec_y + (comp_sec_h - total_boxes_h) / 2

# CPUs — full-width bottom row
dark_box(ax, dk_x, dk_start_y, dk_w, dk_h, "CPUs", fontsize=8.5, zorder=5)

# Accelerators — three equal boxes side by side in the upper row
accel_gap = 0.08
accel_w = (dk_w - 2 * accel_gap) / 3
for j, name in enumerate(["A100", "GH200", "MI250X"]):
    dark_box(ax, dk_x + j * (accel_w + accel_gap),
             dk_start_y + dk_h + dk_gap, accel_w, dk_h, name, fontsize=8.0, zorder=5)

# Sub-section: Interconnects (dashed border)
ic_sec_x = comp_sec_x + comp_sec_w + 0.22
ic_sec_y = comp_sec_y
ic_sec_w = W_TOTAL - comp_sec_w - 0.22 - 0.15 - 0.15
ic_sec_h = comp_sec_h
ic_box = FancyBboxPatch(
    (ic_sec_x, ic_sec_y), ic_sec_w, ic_sec_h,
    boxstyle="round,pad=0,rounding_size=0.2",
    facecolor=TAN_BG, edgecolor=BORDER, linewidth=1.2,
    linestyle="--", zorder=3,
)
ax.add_patch(ic_box)
# "Interconnects" label outside (above) the dashed box, centered
ax.text(ic_sec_x + ic_sec_w / 2, L4_Y + L4_H - 0.30,
        "Interconnects", ha="center", va="center",
        fontsize=10, fontweight="bold", zorder=4)

ic_items = ["NVLink", "Slingshot", "InfiniBand"]
ic_dk_w = ic_sec_w - 1.60
ic_dk_x = ic_sec_x + 0.80
for i, name in enumerate(ic_items):
    dark_box(ax, ic_dk_x, ic_sec_y + 0.22 + i * (dk_h + 0.12),
             ic_dk_w, dk_h, name, fontsize=8.5, zorder=5)

# ── Save ───────────────────────────────────────────────────────────────────────
out = "docs/assets/hpc-ai-stack.png"
plt.tight_layout(pad=0)
plt.savefig(out, dpi=180, bbox_inches="tight", facecolor="white")
print(f"Saved: {out}")
