"""All six snapshot FD checks before any new native target solve."""
from pathlib import Path
import json,sys,hashlib
from types import SimpleNamespace
import numpy as np
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
sys.path.insert(0,str(ROOT))
from world_rows import fast,verify_fd,convert_rows
from mcr_sim.vessel_assets import load_signed_distance_grid
class SavedAdapter(fast.LiveBeamSDFAdapter):
    def __init__(self,meta):
        self.meta=meta
        self.env=SimpleNamespace(sdf_grid=load_signed_distance_grid(meta['sdf_path']),asset_T_env_sim=np.asarray(meta['asset_T_env_sim']),asset_offset_sim=np.asarray(meta['asset_offset_sim']),asset_source_to_sim_scale=meta['asset_source_to_sim_scale'],catheter_radius=meta['catheter_radius'])
    def _geometry(self):return self.meta['geometry_free']
    def solver_geometry(self):return self._geometry()
def main():
    result={}
    for c in json.loads((HERE/'cases.json').read_text()):
        assert hashlib.sha256(Path(c['snapshot']).read_bytes()).hexdigest()==c['snapshot_sha256']
        meta=json.loads(Path(c['metadata']).read_text());adapter=SavedAdapter(meta)
        with np.load(c['snapshot'],allow_pickle=False) as z:
            q=np.array(z['q_free']);prev=np.array(z['q_prev']);expected=np.array(z['free_clearance'])
        points,profile=fast._profile(adapter,q)
        assert np.max(np.abs(profile-expected))<1e-14
        rows=fast.build_rows_fast_local_fd(adapter=adapter,q_prev=prev,q_free=q,dense_points=points,dense_clearance=profile)
        r=verify_fd(adapter,q,rows);result[c['id']]=r
        assert r['pass'] and r['translation_unchanged'],c['id']
        try:convert_rows(convert_rows(rows,q),q)
        except RuntimeError:pass
        else:raise AssertionError('double conversion guard absent')
    (HERE/'results/independent_fd.json').write_text(json.dumps(result,indent=2)+'\n')
    print('INDEPENDENT_FD_PASS',len(result),'original max',max(r['original_max_error'] for r in result.values()),'corrected max',max(r['corrected_max_error'] for r in result.values()))
if __name__=='__main__':main()
