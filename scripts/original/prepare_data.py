#!/usr/bin/env python3
"""Prepare shared CropAndWeed manifests without decoding label pixels."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from researchos.portable_data import main

if __name__ == "__main__":
    raise SystemExit(main())
