"""What the segmentation model is actually given at inference, step by step, for one saved live frame.

    python experiments/2026-11-09_seg_objects_v1_test/inference_input.py [tag, default edge_a_light2]

Follows live_detect exactly: raw camera frame -> rectify (top-down, 10 px/mm) -> white balance with the
plate brought to the training level -> plate mask (off-plate painted plate grey) -> greyscale copied
into three channels -> the 1024 px tile(s) cut from the work area -> ultralytics' own preprocessing,
captured from the predictor itself (model.predictor.preprocess), which is the tensor the network
runs on. Writes inference_input.jpg (the steps) and inference_tensor.png (the tensor, as an image,
at full resolution) and prints the tensor's shape, type and range.
"""
import sys
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from robot_offline import config
from robot_offline.inf import tiling
from robot_offline.pre import rectify

HERE = Path(__file__).parent
LIVE = HERE / "live"
WEIGHTS = config.ROOT / "models" / "seg_grey_v2" / "best.pt"


def label(img, text, size=900):
    s = size / max(img.shape[:2])
    out = cv2.resize(img, (int(img.shape[1] * s), int(img.shape[0] * s)), interpolation=cv2.INTER_AREA)
    out = cv2.copyMakeBorder(out, 44, size - out.shape[0], 0, size - out.shape[1], cv2.BORDER_CONSTANT,
                             value=(255, 255, 255))
    cv2.putText(out, text, (8, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2, cv2.LINE_AA)
    return out


def main():
    tag = sys.argv[1] if len(sys.argv) > 1 else "edge_a_light2"
    ds = config.load("dataset")
    area, tc = ds["work_area"], ds["tiles"]
    work = (area["x0"], area["y0"], area["width"], area["height"])
    frame = cv2.imread(str(LIVE / f"{tag}_raw.png"))
    cal = rectify.Calibration.load(LIVE / f"{tag}_rectify.json")

    plain = rectify.rectify(frame, cal)                                   # 1. top-down
    balanced, _ = rectify.white_balance(plain, cal.settings, rectify.MODEL_PLATE_LEVEL)   # 2.
    view = rectify.mask_outside_plate(balanced, cal.settings)             # 3. = model_input(...)[0]
    grey = cv2.cvtColor(cv2.cvtColor(view, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)       # 4.
    tiles = tiling.tiles(work, (grey.shape[1], grey.shape[0]), tc["size"], tc["overlap"])
    crops = [np.ascontiguousarray(grey[t.window]) for t in tiles]         # 5.
    model = YOLO(str(WEIGHTS))
    model.predict(crops, imgsz=tc["size"], conf=0.5, retina_masks=True, verbose=False)
    tensor = model.predictor.preprocess(crops)                            # 6. what the network gets
    print(f"frame {frame.shape[1]} x {frame.shape[0]} -> rectified {plain.shape[1]} x {plain.shape[0]} -> "
          f"{len(tiles)} tile(s): " + ", ".join(f"x {t.x0}..{t.x0 + t.size}, y {t.y0}..{t.y0 + t.size}" for t in tiles))
    print(f"tensor: shape {tuple(tensor.shape)} (batch, channels, height, width), {tensor.dtype}, "
          f"range {float(tensor.min()):.3f} to {float(tensor.max()):.3f}, device {tensor.device}")
    t_img = (tensor[0].float().cpu().numpy().transpose(1, 2, 0) * 255).round().astype(np.uint8)[:, :, ::-1]
    same = [np.abs(t_img[:, :, c].astype(int) - t_img[:, :, 0]).max() for c in range(3)]
    print(f"the three channels differ by at most {max(same)} grey levels (grey copied into all three)")
    cv2.imwrite(str(HERE / "inference_tensor.png"), t_img)

    shown_tiles = view.copy()
    for t in tiles:
        cv2.rectangle(shown_tiles, (t.x0, t.y0), (t.x0 + t.size - 1, t.y0 + t.size - 1), (0, 0, 255), 3)
    steps = [
        label(frame, f"0. camera frame, {frame.shape[1]} x {frame.shape[0]}"),
        label(plain, "1. rectified top-down, 10 px/mm, 1200 x 1200"),
        label(balanced, "2. white-balanced, plate at the training level"),
        label(view, "3. plate mask: off-plate painted plate grey"),
        label(shown_tiles, f"4. tile(s) cut from the work area (red), {tc['size']} px"),
        label(t_img, f"5. the tensor: {tuple(tensor.shape)}, grey x 3, /255"),
    ]
    sheet = np.vstack([np.hstack(steps[:3]), np.hstack(steps[3:])])
    cv2.imwrite(str(HERE / "inference_input.jpg"), sheet, [cv2.IMWRITE_JPEG_QUALITY, 88])


if __name__ == "__main__":
    main()
