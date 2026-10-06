"""Grabs full-resolution frames from the overhead camera into data/captures/<session>/.

Opens the camera with DirectShow in MJPG at 4608 × 2592, locks focus, exposure and white balance
from configs/camera.toml, checks the frame shape, and saves lossless PNG. Also writes session.toml
and captures empty-table frames into data/backgrounds/<session>/.
"""
