"""Figures that explain the close-up method, from a real timed cycle (session 2026-11-09_f).

    python experiments/2026-11-09_band_focus/make_method_figures.py experiments/2026-11-09_band_focus
"""

import json
import sys
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import ConnectionPatch, Rectangle  # noqa: E402
from ultralytics import YOLO  # noqa: E402

from robot_offline import config  # noqa: E402
from robot_offline.inf import tiling  # noqa: E402
from robot_offline.post.bands import closeup  # noqa: E402
from robot_offline.pre import rectify  # noqa: E402

D = Path(sys.argv[1])
FIG = D / "figures"
FIG.mkdir(exist_ok=True)
S = config.DATA_DIR / "captures" / "2026-11-09_f"
run = json.load(open(S / "predictions.json"))
cal = rectify.Calibration.load(S / "rectify.json")
st = cal.settings
model = closeup.FocusModel(config.load("camera")["closeup"]["points"])
CONFIG_FOCUS = config.load("camera")["camera"]["focus"]
NAMES = {1: "beige 1 kΩ", 2: "green 10 kΩ", 3: "blue 4.7 kΩ", 4: "turquoise"}
TRUTH = {1: 1000.0, 2: 10000.0, 3: 4700.0}         # the turquoise's value isn't confirmed
INK, MUTED, GRID, RED = "#0b0b0b", "#52514e", "#e4e3df", "#b3261e"
FRONT, BACK = "#eb6834", "#2a78d6"     # categorical slots 2 and 1: front-of-plate and back-of-plate resistors
NL = "\n"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.facecolor": "white"})
bodies = run["bodies"]
lens = closeup.lens_mm(cal)
plain = rectify.rectify(cv2.imread(str(S / "0001.png")), cal)
balanced, _ = rectify.white_balance(plain, st)
side = {b["body"]: (FRONT if b["closeup"]["focus"] > CONFIG_FOCUS else BACK) for b in bodies}

# The outlines (read_plate keeps only centres): segment the first capture again.
view, _ = rectify.model_input(cv2.imread(str(S / "0001.png")), cal, plain)
w = config.load("dataset")["work_area"]
tc = config.load("dataset")["tiles"]
polys, _ = tiling.predict_frame(YOLO(str(config.ROOT / "models/seg_grey_v2/best.pt")), view,
                                tiling.tiles((w["x0"], w["y0"], w["width"], w["height"]), view.shape[1::-1],
                                             tc["size"], tc["overlap"]), conf=0.5, grey=True)
outline = {}
for b in bodies:
    d = [np.linalg.norm(np.asarray(p).mean(0) - b["centre"]) for p in polys]
    outline[b["body"]] = np.asarray(polys[int(np.argmin(d))])

# 10. Why one focus can't suit the plate: the rig from the side, the plate shaded by the focus it needs.
fig = plt.figure(figsize=(12, 6.2))
ax = fig.add_axes([0.05, 0.1, 0.58, 0.78])
ax.set_axisbelow(True)
xs = st.plate_corners_mm[:, 0]
x0, x1, z = xs.min(), xs.max(), st.top_z_mm
yc = st.plate_corners_mm[:, 1].mean()
STRIP = 14.0                # mm: the focus strip drawn under the plate, thicker than the plate so it shows


def dist_at(x):
    return float(np.linalg.norm(np.array([x, yc, z]) - lens))


xx = np.linspace(x0, x1, 200)
ff = np.array([model.a + model.b / dist_at(x) for x in xx])
ax.add_patch(Rectangle((x0, z - st.face_mm), x1 - x0, st.face_mm, color="#bdbab2", zorder=2))
ax.imshow(ff[None, :], extent=(x0, x1, z - st.face_mm - 4 - STRIP, z - st.face_mm - 4), aspect="auto", cmap="Blues",
          vmin=255, vmax=335, zorder=1)
ax.text(x0 - 3, z - st.face_mm - 4 - STRIP / 2, "focus\nneeded", ha="right", va="center", fontsize=8, color=MUTED)
for x, f in ((x0, ff[0]), (x1, ff[-1])):
    ax.text(x, z - st.face_mm - 6 - STRIP, f"{f:.0f}", ha="center", va="top", fontsize=8, color=MUTED)
ax.plot(lens[0], lens[2], marker="o", markersize=12, color=INK, zorder=4)
ax.text(lens[0], lens[2] + 7, "lens", ha="center", color=INK)
for x, txt in ((x0, "front edge"), (x1, "back edge")):
    ax.plot([lens[0], x], [lens[2], z], color=MUTED, linewidth=1, linestyle="--")
    mx, mz = (lens[0] + x) / 2, (lens[2] + z) / 2
    ax.text(mx + (8 if x == x1 else -8), mz, f"{txt}: {dist_at(x):.0f} mm", ha="left" if x == x1 else "right",
            va="center", fontsize=8, color=MUTED)
x306 = xx[np.argmin(np.abs(ff - CONFIG_FOCUS))]
ax.plot([x306, x306], [z - st.face_mm - 4 - STRIP, z + 45], color=INK, linewidth=1, linestyle=":")
ax.text(x306, z + 47, f"focus {CONFIG_FOCUS} (the config's)\nis sharpest only here", ha="center", va="bottom",
        fontsize=8, color=INK)
placed = sorted(((rectify.rect_to_mm(np.array([b["centre"]], float), st)[0][0], b) for b in bodies),
                key=lambda t: t[0])
for rank, (x, b) in enumerate(placed):
    up = 14 + 11 * (rank % 2)               # neighbours alternate in height, so their labels don't collide
    ax.plot(x, z + 1.5, marker="v", markersize=10, color=side[b["body"]], markeredgecolor="white", zorder=5)
    ax.plot([x, x], [z + 4, z + up - 1], color=side[b["body"]], linewidth=0.8)
    ax.text(x, z + up, f"#{b['body']}", ha="center", fontsize=9, color=side[b["body"]], fontweight="bold")
ax.set_aspect("equal")
ax.set_xlim(lens[0] - 15, x1 + 15)
ax.set_ylim(z - st.face_mm - 4 - STRIP - 14, lens[2] + 20)
ax.set_xlabel("scene x (mm); the lens is beyond the plate's front edge, on the left")
ax.set_ylabel("height (mm)")
ax.grid(True, color=GRID, linewidth=0.8)
key = NL.join(f"#{b['body']} {NAMES[b['body']]}{NL}     {b['closeup']['distance_mm']:.0f} mm from the lens, "
              f"close-up at focus {b['closeup']['focus']}" for b in sorted(bodies, key=lambda b: b["closeup"]["distance_mm"]))
fig.text(0.6, 0.5, "The resistors on the plate (triangles):" + NL + NL + key + NL + NL +
         "orange: in front of where 306 is sharp," + NL + "so their close-up focus is higher" + NL +
         "blue: behind it, so lower", fontsize=9, color=INK, va="center")
fig.suptitle(f"The rig from the side: the plate is {dist_at(x0):.0f} to {dist_at(x1):.0f} mm from the lens, "
             "so its front and back need different focus", x=0.01, ha="left", fontsize=11, color=INK)
fig.savefig(FIG / "10_how_geometry.png", dpi=130)
plt.close(fig)

# 11. The focus each place on the plate gets, and the resistors on it.
h, wpx = balanced.shape[:2]
uu, vv = np.meshgrid(np.arange(0, wpx, 6), np.arange(0, h, 6))
mm = rectify.rect_to_mm(np.column_stack([uu.ravel(), vv.ravel()]).astype(float), st)
dist = np.linalg.norm(np.column_stack([mm, np.full(len(mm), st.top_z_mm)]) - lens, axis=1).reshape(uu.shape)
focus = model.a + model.b / dist
fig, ax = plt.subplots(figsize=(7.6, 7.2))
ax.imshow(cv2.cvtColor(balanced, cv2.COLOR_BGR2RGB))
cs = ax.contour(uu, vv, focus, levels=np.arange(292, 334, 4), cmap="Blues", linewidths=1.4, vmin=255, vmax=335)
ax.clabel(cs, fmt="%d", fontsize=8, inline=True)
ax.contour(uu, vv, focus, levels=[CONFIG_FOCUS], colors=[MUTED], linestyles=":", linewidths=2)
for b in bodies:
    p = outline[b["body"]]
    ax.fill(p[:, 0], p[:, 1], facecolor="none", edgecolor=side[b["body"]], linewidth=2.5)
    cx, cy = p.mean(0)
    ax.text(cx, cy - 45, f"#{b['body']}: focus {b['closeup']['focus']}", ha="center", fontsize=9, color=INK,
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="none", alpha=0.85))
ax.set_xticks([]), ax.set_yticks([])
ax.set_title(f"Focus by place on the plate (lines every 4 settings; dotted: the config's {CONFIG_FOCUS}){NL}"
             "orange: close-up focus above the config's (front), blue: below (back)", loc="left", fontsize=10, color=INK)
fig.tight_layout()
fig.savefig(FIG / "11_how_focus_map.png", dpi=130)
plt.close(fig)

# 12. One cycle, step by step, for a front and a back resistor.
picks = [next(b for b in bodies if side[b["body"]] == FRONT), next(b for b in bodies if side[b["body"]] == BACK)]
fig = plt.figure(figsize=(14, 6.6))
grid = fig.add_gridspec(2, 5, width_ratios=[1.25, 1.2, 0.95, 1.2, 1.05], wspace=0.3, hspace=0.35, top=0.82)
steps = [f"1. first capture, segmented{NL}(focus 306, 8 frames averaged)", f"2. its crop from{NL}that capture",
         f"3. its distance from the{NL}lens gives its focus",
         f"4. its close-up: focus set, 7 frames{NL}to settle, 4 averaged, found again", "5. read from each"]
rows = []
for b in picks:
    n, c = b["body"], side[b["body"]]
    axes = [fig.add_subplot(grid[len(rows), k]) for k in range(5)]
    rows.append(axes)
    axes[0].imshow(cv2.cvtColor(balanced, cv2.COLOR_BGR2RGB))
    p = outline[n]
    (bx0, by0), (bx1, by1) = p.min(0) - 25, p.max(0) + 25
    axes[0].add_patch(Rectangle((bx0, by0), bx1 - bx0, by1 - by0, fill=False, edgecolor=c, linewidth=2.5))
    axes[1].imshow(cv2.cvtColor(cv2.imread(str(S / "crops" / f"body{n}.png")), cv2.COLOR_BGR2RGB))
    axes[3].imshow(cv2.cvtColor(cv2.imread(str(S / "closeups" / f"body{n}.png")), cv2.COLOR_BGR2RGB))
    k = b["closeup"]
    axes[2].text(0.5, 0.5, f"{k['distance_mm']:.0f} mm{NL}{NL}{model.a:.0f} + {model.b:,.0f} / {k['distance_mm']:.0f}"
                           f"{NL}= focus {k['focus']}", ha="center", va="center", fontsize=9, color=INK,
                 bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor=c, linewidth=2))
    for y, which, lab, best, sharp in ((0.7, "first", b["first_label"], b["best_guess_ohm"], f"{b['sharpness']:,}"),
                                       (0.25, "close-up", b["closeup_label"], k["best_guess_ohm"],
                                        f"{k['sharpness']:,}  (x{k['sharpness'] / b['sharpness']:.2f})")):
        wrong = n in TRUTH and best != TRUTH[n]
        answered = not lab.startswith("?")
        note = f"{NL}WRONG, and over the 0.95 limit" if wrong and answered else (f"{NL}wrong guess, held back" if wrong else "")
        axes[4].text(0.0, y, f"{which}: {lab}{note}{NL}sharpness {sharp}", fontsize=9, va="center",
                     color=RED if wrong and answered else INK)
    for a in axes:
        a.set_xticks([]), a.set_yticks([])
    axes[2].axis("off"), axes[4].axis("off")
    axes[0].set_ylabel(f"#{n} {NAMES[n]}{NL}({'front' if c == FRONT else 'back'} of the plate)", color=INK)
    axes[1].set_title(f"focus {CONFIG_FOCUS}", fontsize=9, color=MUTED)
    axes[3].set_title(f"focus {k['focus']}", fontsize=9, color=MUTED)
    for a, bnext in zip(axes[:-1], axes[1:]):
        fig.add_artist(ConnectionPatch((1.0, 0.5), (0.0, 0.5), "axes fraction", "axes fraction", axesA=a, axesB=bnext,
                                       arrowstyle="-|>", color=MUTED, linewidth=1.2, mutation_scale=14,
                                       shrinkA=6, shrinkB=14))
for a, label in zip(rows[0], steps):
    box = a.get_position()
    fig.text(box.x0 + box.width / 2, 0.86, label, ha="center", va="bottom", fontsize=9, color=INK)
fig.suptitle("One cycle, step by step (session 2026-11-09_f): a resistor at the front of the plate and one at the back",
             x=0.01, y=0.98, ha="left", fontsize=12, color=INK)
fig.savefig(FIG / "12_how_one_cycle.png", dpi=130, bbox_inches="tight")
plt.close(fig)

# 13. Where the time goes: the close-ups of one cycle, measured.
STEP_COLOURS = {"settle": "#2a78d6", "average": "#eb6834", "balance": "#1baf7a", "find_again": "#eda100", "cut": "#e87ba4"}
STEP_NAMES = {"settle": "focus set, 7 frames to settle", "average": "4 frames read and averaged",
              "balance": "white balance", "find_again": "found again (segmentation, tiles round it)", "cut": "crop cut"}
fig, ax = plt.subplots(figsize=(11, 3.6))
order = sorted(bodies, key=lambda b: b["closeup"]["order"])
t0 = 0.0
for row, b in enumerate(order):
    k = b["closeup"]
    for step, secs in k["seconds"].items():
        ax.barh(row, secs, left=t0, height=0.55, color=STEP_COLOURS[step], edgecolor="white", linewidth=2,
                label=STEP_NAMES[step] if row == 0 else None)
        t0 += secs
    ax.text(t0 + 0.1, row, f"{sum(k['seconds'].values()):.1f} s", va="center", fontsize=8, color=MUTED)
ax.set_yticks(range(len(order)), [f"#{b['body']} {NAMES[b['body']]}, focus {b['closeup']['focus']}" for b in order],
              fontsize=8)
ax.invert_yaxis()
ax.set_xlabel("seconds into the close-ups (taken in order of focus, so the lens moves one way)")
ax.grid(True, axis="x", color=GRID, linewidth=0.8)
ax.legend(frameon=False, fontsize=8, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.32))
first = run["seconds"]
ax.set_title(f"Where the time goes: {len(order)} close-ups in {t0:.1f} s, about {t0 / len(order):.1f} s each "
             f"(the first capture itself took {first['first capture (8 frames)']:.1f} s)", loc="left", fontsize=11,
             color=INK)
fig.tight_layout()
fig.savefig(FIG / "13_how_timeline.png", dpi=130)
plt.close(fig)
print("figures in", FIG)
