#!/usr/bin/env python3
"""Prepare, train, and predict with the original PhenoBench native arms."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from researchos.portable_phenobench import main

if __name__ == "__main__":
    raise SystemExit(main())
