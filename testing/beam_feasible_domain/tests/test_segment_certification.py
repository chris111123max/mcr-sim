#!/usr/bin/env python3
"""Adversarial beam segment: endpoints feasible, bowed interior infeasible."""
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from beam_feasible_poc import R,L,cert,controls,bezier

root=Path(__file__).resolve().parents[1]
q=np.array([[0,0,R+.001,0,0,0,1],
            [L,0,R+.001,0,0,0,1],
            [2*L,0,R+.001,0,0,0,1]],float)
q[0,3:7]=Rotation.from_euler("y",90,degrees=True).as_quat()
q[1,3:7]=Rotation.from_euler("y",-90,degrees=True).as_quat()
c=controls(q,0)
endpoints=[float(c[i,2]-R) for i in (0,3)]
mid=float(bezier(c,.5)[2]-R)
out=cert(q)
result={"endpoint_clearance_m":endpoints,"midpoint_clearance_m":mid,
        "adaptive_subdivisions":out["subdivisions"],
        "unresolved_segments":out["unresolved_segments"],
        "max_unresolved_segment_m":out["max_unresolved_segment_m"],
        "certified_feasible":out["feasible"],
        "pass":all(x>0 for x in endpoints) and mid<0 and not out["feasible"] and out["subdivisions"]>0}
path=root/"_runtime/results/beam_segment_certificate.json"
path.parent.mkdir(parents=True,exist_ok=True)
path.write_text(json.dumps(result,indent=2,allow_nan=False))
print(json.dumps(result))
if not result["pass"]: raise SystemExit(1)
