#!/usr/bin/env python3
"""Run only the protected B02 step949/substep1 strict-margin validation."""
from pathlib import Path
import sys
from b02_full_episode_strict_margin_0p100_acceptance import main

if __name__ == "__main__":
    out = Path(__file__).resolve().parents[1] / "_runtime" / "strict_margin_0p100" / "results" / "strict_margin_solver_step949_attempt2.json"
    sys.argv = [sys.argv[0], "--step949-validation", "--max-rl-steps", "949", "--progress-every", "25", "--output", str(out)]
    main()
