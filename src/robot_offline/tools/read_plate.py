"""The whole chain on the plate as it stands: capture, segment, read. Saves predictions before anything
is measured, in data/captures/<session>/predictions.json.

    python -m robot_offline.tools.read_plate <session> <segmentation weights> <reader weights> <picture.jpg> [--closeups]
        [--tagger models/resistor_tag_v1/resistor_tag.pt]

The plate is found, the exposure set for reading (learned.READING_LEVEL), 8 frames averaged at the config
focus, segmented and every body read from that capture. With --closeups each body is then also read from
a capture of its own at the focus that suits it (post/bands/closeup.py), and both readings are kept,
with each crop: <session>/crops/ (the first capture's) and <session>/closeups/, and closeups.jpg side by
side. Greyscale or colour input follows the weights' handoff.json (tiling.wants_grey).

Each body is first tagged resistor, not or unsure (post/resistor_tag.py, --tagger), since an object model
outlines screws, diodes and the rest too. Every body is still read, so the reader's output on
non-resistors can be checked, but a value counts ("trusted") only on a body tagged "resistor" with the
reader above its limit. "unsure" is read but never trusted (the robot measures it), and "not" gets no
close-up. Without the tagger's weights every body counts as a resistor, as before.
"""
import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from robot_offline import config
from robot_offline.inf import tiling
from robot_offline.post import resistor_tag
from robot_offline.post.bands import closeup, crops, learned
from robot_offline.pre import exposure, plate_search, rectify
from robot_offline.pre.camera import open_camera


def reading(r):
    return dict(value_ohm=r.value, best_guess_ohm=r.best, confidence=round(float(r.confidence), 3), bands=list(r.colours))


def label(r, tag=None):
    text = f"{learned.ohms_text(r.value)} {r.confidence:.2f}" if r.value else f"? {learned.ohms_text(r.best)} {r.confidence:.2f}"
    return text if tag in (None, "resistor") else f"[{tag}] {text}"


def trusted(r, tag):
    """Whether a reading's value may be used without measuring: the reader answered, on a resistor."""
    return r.value is not None and tag in (None, "resistor")


def side_by_side(rows, first, close):
    """Each body: its crop from the first capture and from its close-up, with what was read from each."""
    cells = []
    for row, a, b in zip(rows, first, close):
        pair = []
        for img, text in ((a, f"#{row['body']} focus {row['focus']}: {row['first_label']}"),
                          (b, f"close-up, focus {row['closeup']['focus']}: {row['closeup_label']}")):
            tile = cv2.resize(img, (360, 120), interpolation=cv2.INTER_AREA)
            tile = cv2.copyMakeBorder(tile, 26, 4, 4, 4, cv2.BORDER_CONSTANT, value=(255, 255, 255))
            cv2.putText(tile, text, (6, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (20, 20, 20), 1, cv2.LINE_AA)
            pair.append(tile)
        cells.append(np.hstack(pair))
    return np.vstack(cells) if cells else np.full((50, 736, 3), 255, np.uint8)


def main(argv=None):
    p = argparse.ArgumentParser(prog="python -m robot_offline.tools.read_plate")
    p.add_argument("session")
    p.add_argument("seg_weights")
    p.add_argument("reader_weights")
    p.add_argument("picture")
    p.add_argument("--closeups", action="store_true", help="also read each body from its own close-up")
    p.add_argument("--last-fit", action="store_true",
                   help="use where the plate was last found (data/plate/last_rectify.json) instead of searching: "
                        "for a rig that hasn't moved, when parts on the plate (a lead over the centre cross) "
                        "stop the search")
    p.add_argument("--tagger", default=str(config.ROOT / "models" / "resistor_tag_v1" / "resistor_tag.pt"),
                   help="resistor-or-not weights; if missing, every body counts as a resistor")
    args = p.parse_args(argv)
    grey = tiling.wants_grey(args.seg_weights)
    out = config.DATA_DIR / "captures" / args.session
    out.mkdir(parents=True, exist_ok=False)
    ds = config.load("dataset")
    camera = config.load("camera")
    a, tc = ds["work_area"], ds["tiles"]
    seg, net = YOLO(args.seg_weights), learned.load(args.reader_weights)
    tagger = resistor_tag.load(args.tagger) if Path(args.tagger).exists() else None
    if tagger is None:
        print(f"no resistor tag at {args.tagger}: every body counts as a resistor", flush=True)

    cap = open_camera(camera)
    ticks = {"start": time.perf_counter()}
    try:
        if args.last_fit:                   # the last fit as it is; only the exposure brought to READING_LEVEL
            tracker = plate_search.Tracker(cap, rectify.Settings.from_config(ds), camera=camera,
                                           cal=rectify.Calibration.load(plate_search.LAST), locked=True,
                                           level=learned.READING_LEVEL)
            end = time.perf_counter() + 30
            while time.perf_counter() < end:
                ok, f = cap.read()
                if ok:
                    tracker.update(f)
                    if tracker.control.settled:
                        break
            tracker.close()
            cal = tracker.cal
        else:
            try:                            # found dim, then the plate brought to READING_LEVEL for the bands
                cal = plate_search.find_now(cap, camera=camera, level=learned.READING_LEVEL)
            except rectify.PlateNotFound as e:
                out.rmdir()
                raise SystemExit(f"plate not found: {e}")
        print("plate found:", plate_search.describe(cal), flush=True)
        ticks["plate found, exposure set"] = time.perf_counter()
        for _ in range(closeup.SETTLE):
            cap.read()
        frame = closeup.mean_frame(cap, 8)
        ticks["first capture (8 frames)"] = time.perf_counter()
        # As set, for telling captures apart later: band crops depend on all three (focus 314 before 2026-11-09).
        settings = dict(focus=camera["camera"]["focus"], exposure=cap.get(cv2.CAP_PROP_EXPOSURE),
                        gain=cap.get(cv2.CAP_PROP_GAIN), frames_averaged=8)
        print(settings, flush=True)
        cv2.imwrite(str(out / "0001.png"), frame)
        cal.save(out / "rectify.json")
        print("plate fit:", cal.fit, "level", exposure.plate_level(frame, cal)[0])

        plain, gains, white = closeup.white_of(cal, frame)
        view, _ = rectify.model_input(frame, cal, plain)
        t = tiling.tiles((a["x0"], a["y0"], a["width"], a["height"]), (view.shape[1], view.shape[0]),
                         tc["size"], tc["overlap"])
        polys, scores = tiling.predict_frame(seg, view, t, conf=0.5, grey=grey)
        ticks["segmentation"] = time.perf_counter()
        tags = resistor_tag.classify(tagger, view, polys) if tagger is not None else [(None, None)] * len(polys)
        ticks["resistor tag"] = time.perf_counter()
        close = {}                          # body index: its Closeup; none for a body tagged "not"
        wanted = [i for i, (_, tag) in enumerate(tags) if tag != "not"]
        if args.closeups and wanted:
            t0 = time.perf_counter()
            taken = closeup.take(cap, cal, [polys[i] for i in wanted], seg=seg, grey=grey, camera=camera)
            close = dict(zip(wanted, taken))
            close_order = sorted(close, key=lambda i: close[i].focus)    # the order they were taken in
            print(f"{len(close)} close-ups in {time.perf_counter() - t0:.1f} s", flush=True)
            ticks["close-ups"] = time.perf_counter()
    finally:
        cap.release()

    vis = view.copy()
    rows, first_crops = [], []
    (out / "crops").mkdir()
    if close:
        (out / "closeups").mkdir()
    for n, (poly, s, (p_res, tag)) in enumerate(zip(polys, scores, tags), start=1):
        c = crops.cut(frame, cal, poly, gains)
        first_crops.append(c.image)
        cv2.imwrite(str(out / "crops" / f"body{n}.png"), c.image)
        r = learned.read(net, c.image, white)
        (cx, cy), (w, h), _ = cv2.minAreaRect(poly.astype(np.float32))
        row = dict(body=n, centre=[round(cx), round(cy)], length_mm=round(max(w, h) / 10, 1),
                   width_mm=round(min(w, h) / 10, 1), detection=round(float(s), 2), focus=settings["focus"],
                   tag=tag, p_resistor=None if p_res is None else round(p_res, 3),
                   sharpness=round(closeup.sharpness(c.image)), **reading(r), trusted=trusted(r, tag),
                   first_label=label(r, tag))
        tagged = "" if tag is None else f" {tag} {p_res:.2f}"
        line = f"{n}: ({cx:.0f},{cy:.0f}) det {s:.2f}{tagged} focus {settings['focus']} -> {label(r, tag)}"
        if n - 1 in close:
            k = close[n - 1]
            rc = learned.read(net, k.crop.image, k.white)
            cv2.imwrite(str(out / "closeups" / f"body{n}.png"), k.crop.image)
            row["closeup"] = dict(focus=k.focus, distance_mm=round(k.distance_mm, 1), found_again=k.found_again,
                                  sharpness=round(k.sharpness), seconds=k.seconds, order=close_order.index(n - 1),
                                  **reading(rc), trusted=trusted(rc, tag))
            row["closeup_label"] = label(rc, tag)
            line += (f" | close-up {k.distance_mm:.0f} mm, focus {k.focus}"
                     f"{'' if k.found_again else ' (not re-found)'} -> {label(rc, tag)}"
                     f"; sharpness {row['sharpness']} -> {round(k.sharpness)}")
        rows.append(row)
        print(line)
        cv2.polylines(vis, [np.round(poly).astype(np.int32)], True, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(vis, f"{n}: {label(r, tag)}", (int(cx) - 40, int(cy) - 34), 0, .6, (0, 0, 200), 2, cv2.LINE_AA)
    (out / "predictions.json").write_text(json.dumps(
        dict(plate_white=white, camera=settings, reader=str(args.reader_weights), segmentation=str(args.seg_weights),
             resistor_tag=str(args.tagger) if tagger is not None else None,
             unsure=list(resistor_tag.UNSURE) if tagger is not None else None,
             closeup_points=camera["closeup"]["points"] if close else None,
             seconds={k: round(b - a, 2) for (_, a), (k, b) in zip(list(ticks.items()), list(ticks.items())[1:])},
             bodies=rows), indent=1))
    cv2.imwrite(args.picture, vis)
    if close:
        order = sorted(close)
        cv2.imwrite(str(out / "closeups.jpg"), side_by_side([rows[i] for i in order], [first_crops[i] for i in order],
                                                            [close[i].crop.image for i in order]))
    print(len(rows), "bodies")


if __name__ == "__main__":
    main()
