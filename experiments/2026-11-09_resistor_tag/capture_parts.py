"""Captures one placement of mixed parts for testing the resistor tag on real parts.

    python experiments/2026-11-09_resistor_tag/capture_parts.py <tag> [--device cpu]
    python experiments/2026-11-09_resistor_tag/capture_parts.py <tag> --kinds "1=resistor 2=diode 3=screw ..."

Opens the camera as live_detect does and finds the plate. Then it takes a frame, segments it with
seg_objects_v1 at conf 0.5 (on the CPU if asked, so a training run can keep the GPU), and numbers
every object found. It writes parts/<tag>_raw.png (LFS), <tag>_rectify.json, <tag>_numbered.jpg
(the top-down view, each object outlined and numbered) and <tag>.json (outlines, scores, and the
resistor tag's P(resistor) for each).

Then name each numbered object with --kinds, one word each, by number; that only edits <tag>.json, so
it can be done without the camera. Kinds: resistor, diode, capacitor, inductor, ic, transistor,
screw, nut, wire, other. Objects left out are counted as missing a name, not as non-resistors.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from robot_offline import config
from robot_offline.inf import tiling
from robot_offline.post import resistor_tag as rt
from robot_offline.pre import plate_search, rectify

HERE = Path(__file__).parent
OUT = HERE / "parts"
SEG = config.ROOT / "models" / "seg_objects_v1" / "best.pt"
TAG = config.ROOT / "models" / "resistor_tag_v1" / "resistor_tag.pt"


def name_kinds(tag, spec):
    meta = OUT / f"{tag}.json"
    rec = json.loads(meta.read_text())
    for item in spec.split():
        n, kind = item.split("=")
        rec["objects"][int(n) - 1]["kind"] = kind
    meta.write_text(json.dumps(rec, indent=1))
    for i, o in enumerate(rec["objects"], 1):
        print(f"  {i:2d} {o.get('kind', '?'):>10}  P(resistor) {o['p_resistor']:.2f} -> {o['tag']}")


def capture(tag, device):
    from robot_offline.pre.camera import open_camera
    ds = config.load("dataset")
    st = rectify.Settings.from_config(ds)
    area, tc = ds["work_area"], ds["tiles"]
    work = (area["x0"], area["y0"], area["width"], area["height"])
    camera = config.load("camera")
    cap = open_camera(camera)
    cal = plate_search.find_now(cap, st, camera=camera)
    for _ in range(6):
        cap.read()
    ok, frame = cap.read()
    cap.release()
    plain = rectify.rectify(frame, cal)
    view, _ = rectify.model_input(frame, cal, plain)
    tiles = tiling.tiles(work, (view.shape[1], view.shape[0]), tc["size"], tc["overlap"])
    polys, scores = tiling.predict_frame(YOLO(str(SEG)), view, tiles, conf=0.5, grey=tiling.wants_grey(SEG), device=device)
    order = sorted(range(len(polys)), key=lambda i: (round(np.mean(polys[i], 0)[1] / 150), np.mean(polys[i], 0)[0]))
    polys, scores = [polys[i] for i in order], [scores[i] for i in order]
    tags = rt.classify(rt.load(TAG), view, polys) if TAG.exists() else [(float("nan"), "?")] * len(polys)
    OUT.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(OUT / f"{tag}_raw.png"), frame)
    cal.save(OUT / f"{tag}_rectify.json")
    shown = plain.copy()
    for i, (q, (p, t)) in enumerate(zip(polys, tags), 1):
        q = np.round(np.asarray(q)).astype(np.int32)
        colour = {"resistor": (0, 200, 0), "not": (0, 0, 255), "unsure": (0, 200, 255)}.get(t, (255, 0, 255))
        cv2.polylines(shown, [q], True, colour, 2, cv2.LINE_AA)
        x, y = q[:, 0].max() + 6, int(q[:, 1].mean())
        for c, w in (((255, 255, 255), 5), ((0, 0, 0), 2)):
            cv2.putText(shown, str(i), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, c, w, cv2.LINE_AA)
    cv2.imwrite(str(OUT / f"{tag}_numbered.jpg"), shown, [cv2.IMWRITE_JPEG_QUALITY, 90])
    rec = {"tag": tag, "seg": str(SEG.relative_to(config.ROOT)), "resistor_tag": str(TAG.relative_to(config.ROOT)),
           "objects": [{"polygon": np.round(q, 1).tolist(), "score": float(s), "p_resistor": p, "tag": t}
                       for q, s, (p, t) in zip(polys, scores, tags)]}
    (OUT / f"{tag}.json").write_text(json.dumps(rec, indent=1))
    print(f"{len(polys)} objects; numbered view: {OUT / f'{tag}_numbered.jpg'}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("tag")
    p.add_argument("--device", default=None, help="cpu to leave the GPU to a training run")
    p.add_argument("--kinds", help='name the numbered objects: "1=resistor 2=diode ..."')
    a = p.parse_args()
    if a.kinds:
        name_kinds(a.tag, a.kinds)
    else:
        capture(a.tag, a.device)


if __name__ == "__main__":
    main()
