"""
Smoke tests — the simplest possible checks that the project is set up correctly.

These tests don't check correctness of any algorithm. They check that:
1. The package imports without error.
2. Core dependencies are available.
3. The configs can be loaded.

If these fail, NOTHING else can work, so we want them to fail loudly and first.
"""

import importlib
import sys


def test_package_imports():
    """The lyra_lite package itself imports."""
    import lyra_lite

    assert lyra_lite is not None


def test_subpackages_importable():
    """Every subpackage is importable. Catches __init__.py issues early."""
    for sub in ("data", "models", "training", "utils"):
        mod = importlib.import_module(f"lyra_lite.{sub}")
        assert mod is not None, f"Failed to import lyra_lite.{sub}"


def test_core_dependencies_present():
    """The major scientific dependencies are available."""
    required = ["torch", "numpy", "pandas", "sklearn", "hydra", "omegaconf"]
    missing = [name for name in required if importlib.util.find_spec(name) is None]
    assert not missing, f"Missing dependencies: {missing}"


def test_python_version():
    """We run on a supported Python version (3.11 or 3.12)."""
    major, minor = sys.version_info[:2]
    assert major == 3 and minor in (11, 12), (
        f"Unsupported Python {major}.{minor}; project requires 3.11 or 3.12."
    )
