"""Strict JSON for runtime metrics, dataclasses and numpy values."""
import json
from dataclasses import asdict,is_dataclass
from pathlib import Path
import numpy as np

def json_default(value):
    if isinstance(value,np.generic):return value.item()
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,Path):return str(value)
    if is_dataclass(value):return asdict(value)
    raise TypeError(type(value).__name__)

def write_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+".pending")
    temporary.write_text(json.dumps(value,default=json_default,indent=2,allow_nan=False)+"\n")
    temporary.replace(path)
