"""Test-only API shim for the captured real BeamAdapter/SDF geometry."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from b02_step776_adapter_audit import beam_curve
from mcr_sim.sdf_hard_constraint import sample_sdf_clearance_and_outward

RESULTS=Path(__file__).resolve().parents[1]/'_runtime'/'results'
CAPTURE=RESULTS/'b02_step776_capture.json'

class BeamFeasibleAdapter:
    """Expose the prior offline BeamAdapter geometry helper under the hook API."""
    def __init__(self, env, instrument=None):
        self.env=env
        self.instrument=instrument
        self.capture=json.loads(CAPTURE.read_text())
        self.reference_insert=float(self.capture['inserted_length_m'])
        self.interpolation={'interpolation':self.capture['interpolation'],
                            'topology':self.capture['topology']}
        self.last_sampling={'count':0,'spacing_m':None}

    @staticmethod
    def _data_value(component, name):
        data=getattr(component,name,None)
        if data is None:
            return None
        for accessor in ('value','array'):
            try:
                value=getattr(data,accessor)
                value=value() if callable(value) else value
                return np.asarray(value).tolist()
            except Exception:
                pass
        return None

    def _refresh_live_interpolation(self, inserted_length):
        insert=float(inserted_length)
        ic=None
        if self.instrument is not None:
            try:
                ic=self.instrument.getObject('InterpolGuide')
            except Exception:
                pass
        if ic is None:
            if abs(insert-self.reference_insert)>1e-8:
                raise RuntimeError('live WireBeamInterpolation unavailable for changed insertion')
            return

        live=dict(self.capture['interpolation'])
        for name in ('edgeList','lengthList','curvAbsList','DOF0TransformNode0',
                     'DOF1TransformNode1','dofsAndBeamsAligned','straight','radius','vecID'):
            value=self._data_value(ic,name)
            if value is not None:
                live[name]=value
        edge_ids=np.asarray(live.get('edgeList',[]),dtype=np.int64).reshape(-1)
        lengths=np.asarray(live.get('lengthList',[]),dtype=np.float64).reshape(-1)
        if not len(edge_ids) or len(edge_ids)!=len(lengths) or not np.isfinite(lengths).all():
            raise RuntimeError('invalid live WireBeamInterpolation edgeList/lengthList')
        length_sum=float(lengths.sum())
        if abs(length_sum-insert)>1e-6:
            raise RuntimeError(
                f'live BeamAdapter active length mismatch: sum={length_sum:.12g} m, IRC.xtip={insert:.12g} m'
            )
        self.interpolation={'interpolation':live,'topology':self.capture['topology']}
        self.live_inserted_length=insert

    def sample_state(self, state, inserted_length=None, spacing_m=None):
        insert=self.reference_insert if inserted_length is None else float(inserted_length)
        self._refresh_live_interpolation(insert)
        step=.00025 if spacing_m is None else float(spacing_m)
        if not np.isfinite(step) or step<=0:
            raise ValueError('spacing_m must be finite and positive')
        points,_=beam_curve(np.asarray(state,dtype=np.float64),self.interpolation,max_step=step)
        self.last_sampling={'count':int(len(points)),'spacing_m':step,'inserted_length_m':insert,
                            'edge_ids':list(self.interpolation['interpolation']['edgeList']),
                            'active_length_sum_m':float(np.sum(self.interpolation['interpolation']['lengthList']))}
        return points

    def solver_geometry(self):
        insert=float(self.live_inserted_length)
        ip=self.interpolation['interpolation']
        edges=np.asarray(self.interpolation['topology']['edges'],dtype=np.int64)
        specs=[]
        for i,(eid,length) in enumerate(zip(ip['edgeList'],ip['lengthList'])):
            n0,n1=edges[int(eid)]
            length=float(length)
            specs.append({'edge_list_index':int(i),'topology_edge_id':int(eid),
                          'nodes':[int(n0),int(n1)],'rest_length_m':length,
                          'samples':max(2,int(np.ceil(length/.0001))+1),
                          'compressed_linear_fallback':False})
        return {'inserted_length_m':insert,'interpolation':ip,
                'topology':self.interpolation['topology'],'active_elements':specs}

    def query_production_sdf(self, points):
        clearance,outward,valid=sample_sdf_clearance_and_outward(
            np.asarray(points,dtype=np.float64),
            self.env.sdf_grid,
            self.env.asset_T_env_sim,
            self.env.asset_offset_sim,
            self.env.asset_source_to_sim_scale,
            self.env.catheter_radius,
        )
        return clearance,outward,valid
