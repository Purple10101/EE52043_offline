"""Straightened body crops from the raw camera frame: the band reader's input.

A body found in the rectified image (its polygon, rectified px) is cut from the raw camera frame on a
grid laid along the body's long axis, at CROP_PX_PER_MM. The raw frame is sharper than the rectified
image on the near side of the plate (up to 13 px/mm against 10), and sampling it directly keeps that
detail instead of resampling twice. The grid follows the rectified image's orientation (not the
plate's mm axes, whose y points the other way), so crops are never mirrored. Crops are white-balanced
with the frame's plate gains, like the rectified image, and come with the body's mask.

Which end of a crop is which is arbitrary: the band reader decides the reading direction.

    python -m robot_offline.post.bands.crops --sessions 2026-10-23_a 2026-10-23_b

writes data/bands/crops/<session>_<frame>_<n>.png (and _mask.png), data/bands/crops.json and a
sheet of them, from the segmentation model's detections. Values already entered in crops.json are
kept when it is rewritten.
"""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from robot_offline import config
from robot_offline.inf import tiling
from robot_offline.pre import rectify

CROP_PX_PER_MM = 20.0
MARGIN_MM = (0.8, 0.5)      # kept beyond the body: along its axis, and across it


@dataclass
class Crop:
    image: np.ndarray       # H × W × 3 uint8 BGR, the body's axis along the rows' direction (left to right)
    mask: np.ndarray        # H × W uint8, 255 inside the body's outline
    centre_mm: np.ndarray   # (x, y) scene mm on the plate surface
    angle_deg: float        # the body's axis in the rectified image, degrees from +x (image y down)
    length_mm: float
    width_mm: float


def body_frame(poly_rect):
    """(centre, unit axis, unit normal, length, width) of a body outline, all in rectified px.

    The axis is the outline's long axis (from its second moments); the centre is the middle of its
    extent along and across that axis.
    """
    pts = np.asarray(poly_rect, np.float64)
    m = cv2.moments(pts.astype(np.float32))
    c = np.array([m["m10"] / m["m00"], m["m01"] / m["m00"]])
    a = 0.5 * np.arctan2(2 * m["mu11"], m["mu20"] - m["mu02"])
    axis = np.array([np.cos(a), np.sin(a)])
    normal = np.array([-axis[1], axis[0]])
    along, across = (pts - c) @ axis, (pts - c) @ normal
    centre = c + axis * (along.max() + along.min()) / 2 + normal * (across.max() + across.min()) / 2
    return centre, axis, normal, along.max() - along.min(), across.max() - across.min()


def cut(raw, cal, poly_rect, gains=None, px_per_mm=CROP_PX_PER_MM, margin_mm=MARGIN_MM):
    """The body poly_rect (rectified px) cut from the raw frame as a straightened Crop.

    gains: per-channel white-balance gains (B, G, R), as from rectify.white_balance on this frame.
    """
    s = cal.settings.px_per_mm                         # rectified px per mm
    centre, axis, normal, length, width = body_frame(poly_rect)
    half_u = length / 2 + margin_mm[0] * s
    half_v = width / 2 + margin_mm[1] * s
    step = s / px_per_mm                               # rectified px per crop px
    nu, nv = int(np.ceil(2 * half_u / step)), int(np.ceil(2 * half_v / step))
    u = (np.arange(nu) + 0.5) * step - half_u
    v = (np.arange(nv) + 0.5) * step - half_v
    uu, vv = np.meshgrid(u, v)
    rect_pts = centre + uu[..., None] * axis + vv[..., None] * normal
    raw_pts = rectify.to_full(rect_pts.reshape(-1, 2), cal).reshape(nv, nu, 2).astype(np.float32)
    image = cv2.remap(raw, raw_pts[..., 0], raw_pts[..., 1], cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    if gains is not None:
        image = np.clip(image.astype(np.float32) * gains, 0, 255).astype(np.uint8)

    rel = np.asarray(poly_rect, np.float64) - centre
    crop_pts = np.stack([(rel @ axis + half_u) / step - 0.5, (rel @ normal + half_v) / step - 0.5], 1)
    mask = np.zeros((nv, nu), np.uint8)
    cv2.fillPoly(mask, [np.round(crop_pts * 16).astype(np.int32)], 255, cv2.LINE_AA, 4)
    return Crop(image, mask, rectify.rect_to_mm([centre], cal.settings)[0],
                float(np.degrees(np.arctan2(axis[1], axis[0]))), length / s, width / s)


def _sheet(images, per_row=6, cell=(200, 110)):
    tiles = []
    for img in images:
        tile = np.full((cell[1], cell[0], 3), 30, np.uint8)
        scale = min((cell[0] - 8) / img.shape[1], (cell[1] - 8) / img.shape[0])
        small = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        y, x = (cell[1] - small.shape[0]) // 2, (cell[0] - small.shape[1]) // 2
        tile[y:y + small.shape[0], x:x + small.shape[1]] = small
        tiles.append(tile)
    while len(tiles) % per_row:
        tiles.append(np.full((cell[1], cell[0], 3), 30, np.uint8))
    return np.vstack([np.hstack(tiles[i:i + per_row]) for i in range(0, len(tiles), per_row)])


def main(argv=None):
    from ultralytics import YOLO

    from robot_offline.tools.capture import Session

    parser = argparse.ArgumentParser(prog="python -m robot_offline.post.bands.crops",
                                     description="Cut straightened body crops from captured sessions.")
    parser.add_argument("--sessions", nargs="+", required=True)
    parser.add_argument("--weights", type=Path, default=config.ROOT / "models" / "seg_grey_v2" / "best.pt")
    parser.add_argument("--conf", type=float, default=0.5)
    parser.add_argument("--out", type=Path, default=config.DATA_DIR / "bands")
    args = parser.parse_args(argv)

    ds = config.load("dataset")
    area, tiles_cfg = ds["work_area"], ds["tiles"]
    work_area = (area["x0"], area["y0"], area["width"], area["height"])
    model = YOLO(str(args.weights))
    grey = tiling.wants_grey(args.weights)
    crops_dir = args.out / "crops"
    crops_dir.mkdir(parents=True, exist_ok=True)
    index_path = args.out / "crops.json"
    old = {e["file"]: e for e in json.loads(index_path.read_text())} if index_path.exists() else {}

    entries, previews = [], []
    camera = config.load("camera")
    for name in args.sessions:
        session = Session(name, "", "", camera)
        for frame_entry in session.frames:
            raw = cv2.imread(str(session.dir / frame_entry["file"]))
            cal = session.frame_cal(frame_entry)
            plain = rectify.rectify(raw, cal)
            _, gains = rectify.white_balance(plain, cal.settings)   # crops keep the frame's own level
            view, _ = rectify.model_input(raw, cal, plain)
            tiles = tiling.tiles(work_area, (view.shape[1], view.shape[0]), tiles_cfg["size"], tiles_cfg["overlap"])
            polygons, scores = tiling.predict_frame(model, view, tiles, conf=args.conf, grey=grey)
            stem = f"{name}_{Path(frame_entry['file']).stem}"
            for n, (poly, score) in enumerate(zip(polygons, scores), start=1):
                crop = cut(raw, cal, poly, gains)
                file = f"crops/{stem}_{n:02d}.png"
                cv2.imwrite(str(args.out / file), crop.image)
                cv2.imwrite(str(args.out / file.replace(".png", "_mask.png")), crop.mask)
                entries.append({
                    "file": file, "session": name, "frame": frame_entry["file"], "kind": frame_entry["kind"],
                    "body": n, "score": round(float(score), 3),
                    "polygon_rect": np.round(poly, 2).tolist(),
                    "centre_mm": np.round(crop.centre_mm, 2).tolist(), "angle_deg": round(crop.angle_deg, 1),
                    "length_mm": round(crop.length_mm, 2), "width_mm": round(crop.width_mm, 2),
                    # A reference frame knows its value; otherwise keep any value entered by hand.
                    "value_ohm": frame_entry.get("value_ohm", old.get(file, {}).get("value_ohm")),
                })
                previews.append(crop.image)
            print(f"  {stem}: {len(polygons)} bodies", flush=True)
    index_path.write_text(json.dumps(entries, indent=1))
    if previews:
        cv2.imwrite(str(args.out / "crops_sheet.jpg"), _sheet(previews))
    print(f"Wrote {len(entries)} crops to {crops_dir} and {index_path}")


if __name__ == "__main__":
    main()
