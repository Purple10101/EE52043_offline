"""Resistor values for a frame's bodies, in one call: what a robot cycle needs from band reading.

    from robot_offline.post.bands import values
    net = values.load()                                   # models/band_reader_kit_v1, on the CPU
    readings = values.read_frame(net, frame, cal, polygons)
    for r in readings:
        if r.value is not None: ...                       # ohms, at or above the reader's limit
        else: ...                                         # r.best is only a guess: measure it, or set it aside

frame: the raw camera frame (BGR). cal: its rectify.Calibration. polygons: the bodies' outlines in
rectified px, as the segmentation model gives them (tiling.predict_frame). Each body's band crop is
cut from the raw frame, white-balanced, and read as tools/read_plate reads it.

The confidence a value needs is the reader's limit as learned.load sets it:
configs/band_reader.toml [deploy] min_confidence when set (provisionally 0.7 since 2026-11-11), else the
model's own (0.998 for band_reader_kit_v1). The reader decodes only the values it was trained for (the
resistor kit's), so a value outside the sorter's 10 ohm to 10 kohm range can come out:
`in_sorter_range` tells. Read with the plate at learned.READING_LEVEL (grey 205 to 225) for the bands'
sake: plate_search.Tracker(..., level=learned.READING_LEVEL).
"""
from robot_offline import config
from robot_offline.post.bands import closeup, crops, learned

DEFAULT_READER = config.ROOT / "models" / "band_reader_kit_v1" / "band_reader.pt"
SORTER_OHM = (10.0, 10_000.0)       # the sorter's range (docs/classification.md)


def load(path=None):
    """The band reader (DEFAULT_READER unless path), in eval mode on the CPU, with its limit set."""
    return learned.load(path or DEFAULT_READER)


def read_frame(net, frame, cal, polygons, min_confidence=None):
    """A learned.Reading per polygon (rectified px), in order. min_confidence: None for the reader's own."""
    if not len(polygons):
        return []
    _, gains, white = closeup.white_of(cal, frame)
    return [learned.read(net, crops.cut(frame, cal, poly, gains).image, white, min_confidence) for poly in polygons]


def in_sorter_range(ohms):
    return ohms is not None and SORTER_OHM[0] <= ohms <= SORTER_OHM[1]
