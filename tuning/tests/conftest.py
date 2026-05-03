"""Shared pytest fixtures and import path setup."""
import sys
from pathlib import Path

# Make sweep.py importable from tests/
_HERE = Path(__file__).resolve().parent
_TUNING = _HERE.parent
sys.path.insert(0, str(_TUNING))

FIXTURES = _HERE / "fixtures"
