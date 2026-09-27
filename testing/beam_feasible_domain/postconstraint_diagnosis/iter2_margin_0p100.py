#!/usr/bin/env python3
"""Second same-frame robust-margin physical test, requesting 0.100 mm."""
from pathlib import Path
from iter1_robust_margin import RUNTIME, run

if __name__ == "__main__":
    run(
        0.100,
        Path(RUNTIME / "results" / "iter2_margin_0p100.json").resolve(),
        scientific_iteration=2,
    )
