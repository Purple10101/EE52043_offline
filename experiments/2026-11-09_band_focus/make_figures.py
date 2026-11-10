"""Figures for the README, from grid.json (run 2), run1.csv (run 1), crops/ and raw/.

    python experiments/2026-11-09_band_focus/make_figures.py experiments/2026-11-09_band_focus
"""

import csv
import json
import sys
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from robot_offline.post.bands import learned  # noqa: E402

D = Path(sys.argv[1])
FIG = D / "figures"
FIG.mkdir(exist_ok=True)
grid = json.load(open(D / "grid.json"))
FOCUSES, LEVELS, CONFIG_FOCUS = grid["focuses"], [b[0] for b in grid["levels"]], grid["config_focus"]
NAMES = {0: "blue body, 4.7 kΩ (assumed)", 1: "beige body, 1 kΩ (read by eye)"}
SHORT = {0: "blue 4.7 kΩ", 1: "beige 1 kΩ"}
TRUTH = {0: 4700.0, 1: 1000.0}
# Plate level is an ordered magnitude: one hue, light to dark (sequential), never a rainbow.
LEVEL_COLOUR = {170: "#7fb0ec", 205: "#2a78d6", 230: "#123f7a"}
LEVEL_LABEL = {170: "plate grey ~185", 205: "plate grey ~218 (reading level)", 230: "plate grey ~231"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "figure.facecolor": "white"})

recs = grid["records"]


def rec(body, level, focus):
    return next((r for r in recs if r["body"] == body and r["band"][0] == level and r["focus"] == focus), None)


def p_true(r, ohms):
    """The reader's share for the true value, decoded from the saved slot log-probabilities."""
    lp = np.asarray(r["log_probs"])
    rows = np.arange(learned.SLOTS)
    scores = np.concatenate([lp[rows, learned._FORWARD].sum(1), lp[rows, learned._FORWARD[:, ::-1]].sum(1)])
    share = np.exp(scores - scores.max())
    share /= share.sum()
    both = share[:len(learned._OHMS)] + share[len(learned._OHMS):]
    return float(both[learned._OHMS == ohms].sum())


def rgb(path):
    return cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)


# 1. Before and after: the raw frame round each body at the config focus and at its sharpest (322).
fig, axes = plt.subplots(2, 2, figsize=(9, 5.2))
for row, body in enumerate((1, 0)):
    for col, focus in enumerate((CONFIG_FOCUS, 322)):
        ax = axes[row, col]
        ax.imshow(rgb(D / "raw" / f"body{body}_plate205_focus{focus}.png"))
        ax.set_xticks([]), ax.set_yticks([]), ax.grid(False)
        r = rec(body, 205, focus)
        ok = r["best"] == TRUTH[body]
        verdict = f"reads {learned.ohms_text(r['best'])} at {r['confidence']:.2f}" + ("" if ok else "  (wrong)")
        which = "config focus" if focus == CONFIG_FOCUS else "sharpest for these bodies"
        ax.set_title(f"{SHORT[body]}, focus {focus} ({which})\nsharpness {r['sharpness']:,.0f}; {verdict}",
                     fontsize=9, color=INK, loc="left")
fig.suptitle("Before and after: the raw camera frame round each body, plate at the reading level", x=0.01,
             ha="left", fontsize=11, color=INK)
fig.tight_layout()
fig.savefig(FIG / "1_before_after_raw.png", dpi=130)
plt.close(fig)

# 2. The crops the reader is given, at every focus (reading level), with what it read.
fig, axes = plt.subplots(2, len(FOCUSES), figsize=(13, 3.6))
for row, body in enumerate((1, 0)):
    for col, focus in enumerate(FOCUSES):
        ax = axes[row, col]
        ax.imshow(rgb(D / "crops" / f"body{body}_plate205_focus{focus}.png"))
        ax.set_xticks([]), ax.set_yticks([]), ax.grid(False)
        r = rec(body, 205, focus)
        ok = r["best"] == TRUTH[body]
        ax.set_title(f"focus {focus}{' (config)' if focus == CONFIG_FOCUS else ''}\n"
                     f"{learned.ohms_text(r['best'])} {r['confidence']:.2f}{'' if ok else '  WRONG'}",
                     fontsize=9, color=INK if ok else "#b3261e")
        if col == 0:
            ax.set_ylabel(SHORT[body], color=INK)
fig.suptitle("The band crops the reader gets, softest (left) to sharpest (right); reading and confidence above each",
             x=0.01, ha="left", fontsize=11, color=INK)
fig.tight_layout()
fig.savefig(FIG / "2_crops_by_focus.png", dpi=130)
plt.close(fig)


def per_level_lines(ax, body, value, ylabel):
    for level in LEVELS:
        ys = [value(rec(body, level, f)) for f in FOCUSES]
        ax.plot(FOCUSES, ys, color=LEVEL_COLOUR[level], linewidth=2, marker="o", markersize=8,
                markeredgecolor="white", markeredgewidth=2, label=LEVEL_LABEL[level])
    ax.axvline(CONFIG_FOCUS, color=MUTED, linewidth=1, linestyle=":")
    ax.text(CONFIG_FOCUS + 0.5, ax.get_ylim()[1], "config focus", color=MUTED, fontsize=8, va="top")
    ax.set_xticks(FOCUSES)
    ax.set_xlabel("focus setting (higher = focused nearer the front of the plate)")
    ax.set_ylabel(ylabel)
    ax.set_title(NAMES[body], loc="left", fontsize=10, color=INK)


# 3. Sharpness against focus: does focus do what it should?
fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
for ax, body in zip(axes, (1, 0)):
    per_level_lines(ax, body, lambda r: r["sharpness"], "crop sharpness (mean squared gradient)")
axes[0].legend(frameon=False, fontsize=8, loc="upper left")
fig.suptitle("Focus works: both front-of-plate bodies get steadily sharper towards 322", x=0.01, ha="left",
             fontsize=11, color=INK)
fig.tight_layout()
fig.savefig(FIG / "3_sharpness_vs_focus.png", dpi=130)
plt.close(fig)

# 4. The reader's belief in the right value against focus.
fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
for ax, body in zip(axes, (1, 0)):
    ax.set_ylim(-0.03, 1.05)
    per_level_lines(ax, body, lambda r, b=body: p_true(r, TRUTH[b]), "reader's probability for the true value")
    ax.axhline(learned.MIN_CONFIDENCE, color="#b3261e", linewidth=1, linestyle="--",
               label=f"reader answers only above {learned.MIN_CONFIDENCE}")
axes[0].legend(frameon=False, fontsize=8, loc="lower left")
fig.suptitle("The reader does not like sharp crops: the beige 1 kΩ is read best at 298 to 306 and lost by 314 to 322",
             x=0.01, ha="left", fontsize=11, color=INK)
fig.tight_layout()
fig.savefig(FIG / "4_reader_vs_focus.png", dpi=130)
plt.close(fig)

# 5. Both runs: confidence and right or wrong, beige, against sharpness (the variable the reader sees).
run1 = [dict(r, body=int(r["body"]), level=int(r["plate_band"]), focus=int(r["focus"]), best=float(r["best_ohm"]),
             confidence=float(r["confidence"]), sharpness=float(r["sharpness"])) for r in csv.DictReader(open(D / "run1.csv"))]
run2 = [dict(body=r["body"], level=r["band"][0], focus=r["focus"], best=r["best"], confidence=r["confidence"],
             sharpness=r["sharpness"]) for r in recs]
fig, ax = plt.subplots(figsize=(8, 4))
for runs, marker, label in ((run1, "o", "run 1"), (run2, "s", "run 2")):
    beige = [r for r in runs if r["body"] == 1]
    right = [r for r in beige if r["best"] == TRUTH[1]]
    wrong = [r for r in beige if r["best"] != TRUTH[1]]
    ax.scatter([r["sharpness"] for r in right], [r["confidence"] for r in right], s=64, marker=marker,
               color="#2a78d6", edgecolor="white", linewidth=1.5, label=f"{label}: read 1 kΩ (right)")
    ax.scatter([r["sharpness"] for r in wrong], [r["confidence"] for r in wrong], s=72, marker="X",
               color="#b3261e", edgecolor="white", linewidth=1.0, label=f"{label}: wrong value" if wrong else None)
ax.axhline(learned.MIN_CONFIDENCE, color="#b3261e", linewidth=1, linestyle="--")
ax.set_xlabel("crop sharpness")
ax.set_ylabel("reader's confidence in its best guess")
ax.set_ylim(0, 1.05)
ax.legend(frameon=False, fontsize=8, loc="lower left")
ax.set_title("Beige 1 kΩ, both runs, all plate levels: right while soft, wrong once sharp", loc="left", fontsize=11, color=INK)
fig.tight_layout()
fig.savefig(FIG / "5_beige_both_runs.png", dpi=130)
plt.close(fig)

# 6. What changes in the reader's mind: beige slot probabilities, read right (298) and wrong (322).
fig, axes = plt.subplots(1, 2, figsize=(11, 3.4), sharey=True)
for ax, focus in zip(axes, (298, 322)):
    r = rec(1, 205, focus)
    probs = np.exp(np.asarray(r["log_probs"]))
    ax.imshow(probs, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(learned.COLOURS)), learned.COLOURS, rotation=60, fontsize=8)
    ax.set_yticks(range(learned.SLOTS), [f"slot {i + 1}" for i in range(learned.SLOTS)], fontsize=8)
    ax.grid(False)
    for i in range(learned.SLOTS):
        j = int(np.argmax(probs[i]))
        ax.text(j, i, f"{probs[i, j]:.2f}", ha="center", va="center", fontsize=7,
                color="white" if probs[i, j] > 0.5 else INK)
    ax.set_title(f"focus {focus}: reads {learned.ohms_text(r['best'])} ({r['confidence']:.2f}), "
                 f"bands {'-'.join(r['colours'])}", loc="left", fontsize=9, color=INK)
fig.suptitle("Beige 1 kΩ (brown, black, red, gold), plate at the reading level: the reader's colour for each band slot",
             x=0.01, ha="left", fontsize=11, color=INK)
fig.tight_layout()
fig.savefig(FIG / "6_beige_slot_colours.png", dpi=130)
plt.close(fig)
print("figures in", FIG)
