#!/usr/bin/env python3
"""Train, predict, and evaluate the original four systems."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from researchos.portable_original import main

if __name__ == "__main__":
    raise SystemExit(main())
