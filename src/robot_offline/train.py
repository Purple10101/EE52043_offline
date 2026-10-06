"""Fine-tunes a COCO-pretrained YOLO-seg model on data/yolo/ with configs/train.toml.

Output goes to runs/<name>/. Records the tile size, overlap and label rules_version next to the
weights, for the hand-off to the robot repo.
"""
