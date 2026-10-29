import ast
import importlib
from pathlib import Path

import pytest

PACKAGE = Path(__file__).resolve().parents[1] / "src" / "robot_offline"
MODULES = sorted(
    ".".join(("robot_offline",) + p.relative_to(PACKAGE).with_suffix("").parts).removesuffix(".__init__")
    for p in PACKAGE.rglob("*.py")
)


@pytest.mark.parametrize("name", MODULES)
def test_module_imports(name):
    importlib.import_module(name)


@pytest.mark.parametrize("name", [m for m in MODULES if not m.startswith("robot_offline.training")])
def test_nothing_outside_training_imports_training(name):
    path = PACKAGE.joinpath(*name.split(".")[1:])
    path = path / "__init__.py" if path.is_dir() else path.with_suffix(".py")
    imported = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
            imported.update(f"{node.module}.{a.name}" for a in node.names)
        elif isinstance(node, ast.Import):
            imported.update(a.name for a in node.names)
    assert not [m for m in imported if m.startswith("robot_offline.training")]
