#!/usr/bin/env python3
"""Run the pinned S7 prerequisite with the original paired-data adapter."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from researchos.portable_phenobench_s7 import main
if __name__ == "__main__":
    raise SystemExit(main())
