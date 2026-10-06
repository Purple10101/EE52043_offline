import importlib

import pytest

MODULES = [
    "config",
    "manifest",
    "capture",
    "autolabel",
    "cutouts",
    "synth",
    "tiling",
    "export_yolo",
    "train",
    "evaluate",
    "visualise",
]


@pytest.mark.parametrize("name", MODULES)
def test_module_imports(name):
    importlib.import_module(f"robot_offline.{name}")
