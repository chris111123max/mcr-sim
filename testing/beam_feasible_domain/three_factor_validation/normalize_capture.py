#!/usr/bin/env python3
"""Inspect and normalize a pre-existing test-only Beam snapshot (no SOFA)."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

import numpy as np

CANONICAL_REQUIRED = (
    "q_free", "q_committed", "row_offsets", "dof_indices",
    "linear_jacobian", "angular_jacobian", "free_violations",
    "selected_dense_indices", "q_free_dense_clearance",
    "q_committed_dense_clearance",
)
CANONICAL_OPTIONAL = ("q_prev", "angular_world_oracle")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--inspect", action="store_true",
                        help="List fields and shapes; do not write anything")
    parser.add_argument("--mapping", type=Path, help="Canonical -> source key JSON")
    parser.add_argument("--output", type=Path, help="New normalized .npz")
    args = parser.parse_args()

    with np.load(args.source, allow_pickle=False) as npz:
        if args.inspect:
            print(json.dumps(
                {k: {"shape": list(npz[k].shape), "dtype": str(npz[k].dtype)}
                 for k in npz.files}, indent=2, ensure_ascii=False
            ))
            return 0
        if not args.mapping or not args.output:
            parser.error("normalization requires both --mapping and --output")
        mapping = json.loads(args.mapping.read_text(encoding="utf-8"))
        if not isinstance(mapping, dict):
            raise ValueError("Mapping must be JSON object canonical_name -> source_key")
        allowed = set(CANONICAL_REQUIRED + CANONICAL_OPTIONAL)
        if any(key not in allowed for key in mapping):
            raise ValueError("Unknown canonical key in mapping")
        found = {}
        for key in CANONICAL_REQUIRED + CANONICAL_OPTIONAL:
            source = mapping.get(key, key)
            if not isinstance(source, str):
                raise ValueError("Source key must be a string: " + key)
            if source not in npz.files:
                if key in CANONICAL_REQUIRED:
                    raise ValueError("Missing REQUIRED source key " + key + ": " + source)
                continue
            value = np.asarray(npz[source])
            if not np.issubdtype(value.dtype, np.number):
                raise ValueError("Nonnumeric snapshot array: " + key)
            if not np.isfinite(value).all():
                raise ValueError("Nonfinite snapshot array: " + key)
            found[key] = np.array(value)
    src = args.source.resolve()
    dst = args.output.resolve()
    if src == dst:
        raise ValueError("Output must differ from input (do not overwrite captures)")
    if dst.suffix != ".npz":
        raise ValueError("Output must end in .npz")
    dst.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(dst, **found)
    print(json.dumps({
        "status": "NORMALIZED", "source": str(src), "output": str(dst),
        "fields": {k: list(v.shape) for k, v in found.items()}
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError, KeyError) as exc:
        print("NORMALIZATION_BLOCKED: " + str(exc), file=sys.stderr)
        sys.exit(2)
