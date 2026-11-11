"""The band collector (tools/collect_bands.py) for just the three values the band reader needs most
(experiments/2026-11-11_band_reader_kit): 56k, 100k and 1M, in that order, into the 2026-11-10 kit session.

    python -m robot_offline.tools.collect_priority

Together they give training its first orange (100k) and yellow (1M) multipliers and a second red one
(56k, also the only part with blue as its second band). Same window, keys and saving as collect_bands:
5 placements each, about 8 parts a placement; quit with q and run it again to carry on.
"""
from robot_offline.tools import collect_bands

SESSION = "2026-11-10"
VALUES = "56k,100k,1M"


def main(argv=None):
    collect_bands.main(["--session", SESSION, "--only", VALUES] + list(argv or []))


if __name__ == "__main__":
    import sys
    main(sys.argv[1:])
