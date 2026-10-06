"""Loads configs/camera.toml, dataset.toml and train.toml into frozen settings objects.

Fails at load with a message naming the file and key if anything is missing or inconsistent
(for example train imgsz differing from the tile size).
"""
