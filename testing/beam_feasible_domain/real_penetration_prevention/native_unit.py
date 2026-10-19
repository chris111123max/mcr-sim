"""Real native callback unit smoke before saved-state replay; no physics steps."""
from pathlib import Path
import ctypes,json,numpy as np
HERE=Path(__file__).resolve().parent
import Sofa
lib=ctypes.CDLL(str(HERE/'results/native_build/lib/libMCRContactReferenceConsistency.so'),mode=ctypes.RTLD_GLOBAL)
f=lib.referenceContactUnitTest;f.argtypes=[ctypes.POINTER(ctypes.c_double)];f.restype=ctypes.c_int
values=(ctypes.c_double*9)();code=f(values);v=np.asarray(list(values)).reshape(3,3)
result={'native_return':code,'original_free_rhs_m':v[0].tolist(),'selected_free_rhs_m':v[1].tolist(),'selected_current_rhs_m':v[2].tolist(),
        'baseline_exact':bool(np.array_equal(v[0],v[1])),
        'updated_normal_matches_geometry':bool(abs(v[2,0]-0.00005)<1e-15),
        'original_normal_matches_geometry':bool(abs(v[0,0]+0.0001)<1e-15),
        'finite':bool(np.isfinite(v).all()),'physics_substeps':0,'synthetic_initialization_only':True}
result['pass']=code==0 and all(result[k] for k in ('baseline_exact','updated_normal_matches_geometry','original_normal_matches_geometry','finite'))
(HERE/'results/native_unit.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result));assert result['pass']
