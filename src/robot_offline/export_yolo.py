"""Exports the manifest as a YOLO-seg dataset of tiles in data/yolo/.

Writes images/{train,val,test}, labels/{train,val,test} (class 0, normalised polygon per line)
and dataset.yaml. Splits by session from configs/dataset.toml; synthetic frames are train only.
"""
