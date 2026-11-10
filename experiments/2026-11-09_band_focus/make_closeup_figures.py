"""Figures for the close-up method (README, part 2), from the two test cycles' predictions and crops and
the focus points in configs/camera.toml.

    python experiments/2026-11-09_band_focus/make_closeup_figures.py experiments/2026-11-09_band_focus
"""

import json
import sys
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from robot_offline import config  # noqa: E402
from robot_offline.post.bands import closeup  # noqa: E402

D = Path(sys.argv[1])
FIG = D / "figures"
FIG.mkdir(exist_ok=True)
SESSIONS = ("2026-11-09_d", "2026-11-09_e")
CAP = config.DATA_DIR / "captures"
NAMES = {1: "#1 green (10 kΩ)", 2: "#2 beige (1 kΩ)", 3: "#3 blue (4.7 kΩ, assumed)", 4: "#4 turquoise (value unconfirmed)"}
TRUTH = {1: 10000.0, 2: 1000.0, 3: 4700.0}
INK, MUTED, GRID, RED = "#0b0b0b", "#52514e", "#e4e3df", "#b3261e"
FIRST, CLOSE = "#9aa0a6", "#2a78d6"        # the first capture recessive grey, the close-up blue
plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "figure.facecolor": "white"})
runs = {s: json.load(open(CAP / s / "predictions.json")) for s in SESSIONS}

# 7. The focus model: measured best focus against distance from the lens, the fit, and what the cycles used.
points = np.asarray(config.load("camera")["closeup"]["points"], float)
model = closeup.FocusModel(points.tolist())
fig, ax = plt.subplots(figsize=(8, 4.2))
d = np.linspace(225, 285, 100)
ax.plot(d, model.a + model.b / d, color=MUTED, linewidth=2, label=f"fit: focus = {model.a:.0f} + {model.b:,.0f} / distance")
ax.scatter(points[:4, 0], points[:4, 1], s=64, color="#7fb0ec", edgecolor="white", linewidth=1.5, zorder=3,
           label="best focus measured, segmentation's sweep")
ax.scatter(points[4:, 0], points[4:, 1], s=64, color="#123f7a", edgecolor="white", linewidth=1.5, zorder=3,
           label="best focus measured, calibrate on this plate")
used = [(b["closeup"]["distance_mm"], b["closeup"]["focus"]) for r in runs.values() for b in r["bodies"]]
ax.scatter(*zip(*used), s=80, marker="D", facecolor="none", edgecolor=CLOSE, linewidth=2, zorder=4,
           label="focus the close-ups were taken at")
ax.axhline(config.load("camera")["camera"]["focus"], color=MUTED, linewidth=1, linestyle=":",
           label="config focus (one setting for the whole plate)")
ax.set_xlabel("distance from the lens to the body (mm): front of the plate left, back right")
ax.set_ylabel("focus setting")
ax.legend(frameon=False, fontsize=8, loc="upper right")
ax.set_title("Where each resistor's close-up is focused: from its distance, by the fitted model", loc="left",
             fontsize=11, color=INK)
fig.tight_layout()
fig.savefig(FIG / "7_focus_model.png", dpi=130)
plt.close(fig)

# 8. Before and after, the cycle's own crops (session e): the first capture's and each close-up.
run = runs["2026-11-09_e"]
fig, axes = plt.subplots(len(run["bodies"]), 2, figsize=(9, 2.1 * len(run["bodies"])))
for row, b in enumerate(run["bodies"]):
    n = b["body"]
    for col, (folder, focus, lab, conf, best) in enumerate((
            ("crops", b["focus"], b["first_label"], b["confidence"], b["best_guess_ohm"]),
            ("closeups", b["closeup"]["focus"], b["closeup_label"], b["closeup"]["confidence"], b["closeup"]["best_guess_ohm"]))):
        ax = axes[row, col]
        ax.imshow(cv2.cvtColor(cv2.imread(str(CAP / "2026-11-09_e" / folder / f"body{n}.png")), cv2.COLOR_BGR2RGB))
        ax.set_xticks([]), ax.set_yticks([]), ax.grid(False)
        wrong = n in TRUTH and best != TRUTH[n]
        answered = conf >= 0.95
        flag = "  CONFIDENTLY WRONG" if wrong and answered else ("  (wrong guess, held back)" if wrong else "")
        ax.set_title(f"{'first capture' if col == 0 else 'close-up'}, focus {focus}: {lab}{flag}", fontsize=9,
                     loc="left", color=RED if wrong and answered else INK)
        if col == 0:
            ax.set_ylabel(NAMES[n], fontsize=8, color=INK)
fig.suptitle("One cycle (session 2026-11-09_e): each resistor from the first capture (left) and its close-up (right)",
             x=0.01, ha="left", fontsize=11, color=INK)
fig.tight_layout()
fig.savefig(FIG / "8_closeup_crops.png", dpi=130)
plt.close(fig)

# 9. Sharpness, first capture against close-up, both cycles: a dumbbell per resistor and cycle.
fig, ax = plt.subplots(figsize=(8.5, 3.8))
labels, y = [], 0
for s in SESSIONS:
    for b in runs[s]["bodies"]:
        a, c = b["sharpness"], b["closeup"]["sharpness"]
        ax.plot([a, c], [y, y], color=GRID, linewidth=4, solid_capstyle="round", zorder=1)
        ax.scatter([a], [y], s=70, color=FIRST, edgecolor="white", linewidth=1.5, zorder=3,
                   label="first capture (focus 306)" if y == 0 else None)
        ax.scatter([c], [y], s=70, color=CLOSE, edgecolor="white", linewidth=1.5, zorder=3,
                   label="close-up (its own focus)" if y == 0 else None)
        ax.text(max(a, c) + 250, y, f"x{c / a:.2f}  (focus {b['closeup']['focus']}, {b['closeup']['distance_mm']:.0f} mm)",
                va="center", fontsize=8, color=MUTED)
        labels.append(f"{NAMES[b['body']].split(' (')[0]}, {s[-1]}")
        y += 1
ax.set_yticks(range(len(labels)), labels, fontsize=8)
ax.invert_yaxis()
ax.set_xlabel("crop sharpness (mean squared gradient)")
ax.set_xlim(right=ax.get_xlim()[1] + 3500)
ax.legend(frameon=False, fontsize=8, loc="upper right")
ax.set_title("Close-ups sharpen the front of the plate (x1.4 to x1.9); the back is already near its best at 306",
             loc="left", fontsize=11, color=INK)
fig.tight_layout()
fig.savefig(FIG / "9_closeup_sharpness.png", dpi=130)
plt.close(fig)
print("figures in", FIG)
