"""Draft outlines for the non-resistor objects in the real test frames, for rules version 2.

    python experiments/2026-11-09_object_seg_design/label_real_objects.py            # drafts and previews
    python experiments/2026-11-09_object_seg_design/label_real_objects.py --write    # into the manifest

Real frames from before rules version 2 have only their resistors outlined. Each other object (the
screws in 2026-10-23_a frames 10 to 15 and the bolt in 2026-11-06_a) has a box placed by eye below.
OpenCV's GrabCut then separates it from the plate inside the box. A colour threshold was tried first: on
a white plate it took shadows and missed most of each zinc screw.

The wood shaving and chip in frames 14 and 15 are background (docs/labelling.md, rule 8) and get no box.

Drafts are drawn to real_labels/ for checking by eye. With --write, they are added (kind "screw",
label_source "auto") and the frames are marked labelled "objects".

The bolt scene in 2026-11-06_a is one layout at 12 exposures. Its outline is drafted on the brightest
frame and used for all 12.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from robot_offline import config, manifest
from robot_offline.pre import rectify

HERE = Path(__file__).parent
OUT = HERE / "real_labels"
# Boxes (x0, y0, x1, y1, rectified px) round each non-resistor object, read off gridded frames by eye.
BOXES = {
    "captures/2026-10-23_a/rectified/0010.png": [(895, 435, 1055, 570)],
    "captures/2026-10-23_a/rectified/0011.png": [(195, 470, 345, 590)],
    "captures/2026-10-23_a/rectified/0012.png": [(360, 170, 560, 300), (765, 820, 895, 965)],
    "captures/2026-10-23_a/rectified/0013.png": [(195, 290, 420, 400), (830, 255, 975, 365)],
    "captures/2026-10-23_a/rectified/0014.png": [(430, 625, 515, 770), (890, 630, 975, 765), (395, 785, 605, 945)],
    "captures/2026-10-23_a/rectified/0015.png": [(425, 625, 515, 770), (890, 630, 975, 765), (765, 975, 1020, 1085)],
}
BOLT_SCENE = [f"captures/2026-11-06_a/rectified/{i:04d}.png" for i in range(1, 13)]
BOLT_BOX = (375, 495, 495, 555)


def draft(image, box, resistor_polys):
    """The object's outline (N×2 px) inside box, by GrabCut; resistor bodies are marked as not it."""
    x0, y0, x1, y1 = box
    pad = 30
    X0, Y0 = max(x0 - pad, 0), max(y0 - pad, 0)
    crop = image[Y0:y1 + pad, X0:x1 + pad].copy()
    mask = np.full(crop.shape[:2], cv2.GC_BGD, np.uint8)
    mask[y0 - Y0:y1 - Y0, x0 - X0:x1 - X0] = cv2.GC_PR_FGD
    for p in resistor_polys:
        cv2.fillPoly(mask, [np.round(p - (X0, Y0)).astype(np.int32)], cv2.GC_BGD)
    bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
    cv2.grabCut(crop, mask, None, bgd, fgd, 8, cv2.GC_INIT_WITH_MASK)
    fg = np.isin(mask, (cv2.GC_FGD, cv2.GC_PR_FGD)).astype(np.uint8)
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    c = max(contours, key=cv2.contourArea)
    return cv2.approxPolyDP(c, 1.0, True).reshape(-1, 2).astype(float) + (X0, Y0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--write", action="store_true")
    a = p.parse_args()
    ds = config.load("dataset")
    st = rectify.Settings.from_config(ds)
    labels = manifest.load(ds["labels"]["rules_version"])
    by_image = {s["image"]: s for s in labels["samples"]}

    def resistors(sample):
        """The frame's resistor outlines only, kept out of each draft. Screws drafted on an earlier run
        are left out, so running this again redrafts them instead of masking them away."""
        return [np.asarray(i["polygon"], float) for i in sample["instances"] if i.get("kind", "resistor") == "resistor"]
    OUT.mkdir(exist_ok=True)
    drafts = {}
    for key, boxes in BOXES.items():
        img = cv2.imread(str(config.DATA_DIR / key))
        drafts[key] = [draft(img, b, resistors(by_image[key])) for b in boxes]
    bright = max(BOLT_SCENE, key=lambda k: cv2.imread(str(config.DATA_DIR / k)).mean())
    bolt = [draft(cv2.imread(str(config.DATA_DIR / bright)), BOLT_BOX, resistors(by_image[bright]))]
    for key in BOLT_SCENE:
        drafts[key] = bolt
    summary = {}
    for key, polys in drafts.items():
        img = cv2.imread(str(config.DATA_DIR / key))
        for q in manifest.polygons(by_image[key]):
            cv2.polylines(img, [np.round(q).astype(np.int32)], True, (0, 200, 0), 2, cv2.LINE_AA)
        for q in polys:
            cv2.polylines(img, [np.round(q).astype(np.int32)], True, (200, 60, 200), 2, cv2.LINE_AA)
        name = key.split("/")[1] + "_" + Path(key).stem
        cv2.imwrite(str(OUT / f"{name}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 90])
        summary[key] = len(polys)
        print(f"{key}: {len(polys)} objects")
    (OUT / "drafts.json").write_text(json.dumps({k: [manifest.polygon_to_json(q) for q in v] for k, v in drafts.items()},
                                                indent=1))
    # Real frames checked by eye (2026-11-09) to hold no object but their outlined resistors (or nothing,
    # or only things bigger than the pickable size: the arm and servos in 2026-11-05_live). Already complete.
    complete = [s["image"] for s in labels["samples"] if s["source"] == "real" and s["image"] not in drafts
                and s["session"] in COMPLETE_SESSIONS
                and not (s["session"] == "2026-10-23_b" and int(Path(s["image"]).stem) in PENDING_23B)]
    print(f"{len(complete)} frames hold only their outlined resistors")
    if a.write:
        for key, polys in drafts.items():
            s = by_image[key]
            s["instances"] = [i for i in s["instances"] if i.get("kind", "resistor") == "resistor"]
            s["instances"] += [{"polygon": manifest.polygon_to_json(q), "label_source": "auto", "kind": "screw"}
                               for q in polys]
            s["labelled"] = "objects"
        for key in complete:
            by_image[key]["labelled"] = "objects"
        manifest.save(labels)
        print(f"wrote {sum(summary.values())} object outlines into {len(drafts)} frames; "
              f"{len(complete)} more marked complete")


COMPLETE_SESSIONS = ("2026-10-23_a", "2026-10-23_b", "2026-10-29_ref", "2026-11-05_live", "2026-11-06_b", "2026-11-06_c")
PENDING_23B = (13, 14, 15)   # screws, a foil ball and wood: left "resistors" until foil is decided (object or debris)


if __name__ == "__main__":
    main()
