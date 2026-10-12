"""Standalone C++ vs Python coarse-safety parity test (no SOFA runtime needed).

From repo root, within mcr_sofa with SciPy available:
  python training/beam_rhs_consistent/test_native_guard.py

Uses the true SOFA-independent C++ kernel compiled as a temporary shared lib.
The worker also runs three real-SOFA parity checks automatically at startup.
"""
import ctypes
import pathlib
import subprocess
import sys
import tempfile
import numpy as np
from scipy.spatial.transform import Rotation

HERE=pathlib.Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(HERE))
sys.path.insert(0,str(ROOT))
from fast_guard import coarse_beam_clearance_guard


class Grid:
    def __init__(self):
        self.spacing=np.array([0.5,0.5,0.5])
        self.origin=np.array([-16.,-16.,-16.])
        self.values=np.zeros((65,65,65),dtype=np.float32)
        x=self.origin[0]+np.arange(65)*self.spacing[0]
        y=self.origin[1]+np.arange(65)*self.spacing[1]
        z=self.origin[2]+np.arange(65)*self.spacing[2]
        self.values[:]=np.sqrt(y[None,:,None]**2+z[:,None,None]**2)-5.0
    def sample(self,points):
        p=np.asarray(points,dtype=float).reshape(-1,3)
        xyz=(p-self.origin)/self.spacing
        valid=np.all((xyz>=0)&(xyz<=64),axis=1)&np.isfinite(xyz).all(axis=1)
        out=np.full(len(p),np.inf)
        for idx in np.where(valid)[0]:
            q=xyz[idx]; lo=np.clip(np.floor(q).astype(int),0,63);f=np.clip(q-lo,0,1)
            x,y,z=lo;dx,dy,dz=f
            v=self.values
            c00=v[z,y,x]*(1-dx)+v[z,y,x+1]*dx
            c10=v[z,y+1,x]*(1-dx)+v[z,y+1,x+1]*dx
            c01=v[z+1,y,x]*(1-dx)+v[z+1,y,x+1]*dx
            c11=v[z+1,y+1,x]*(1-dx)+v[z+1,y+1,x+1]*dx
            out[idx]=(c00*(1-dy)+c10*dy)*(1-dz)+(c01*(1-dy)+c11*dy)*dz
        return out


class Adapter:
    def __init__(self,grid,pose,offset,lengths,ids):
        self.grid,self.pose,self.offset=grid,pose,offset
        self.lengths,self.ids=lengths,ids
    def _geometry(self):
        return {"active_elements":[{"nodes":list(pair),"rest_length_m":float(l)}
                                   for pair,l in zip(self.ids,self.lengths)]}
    def query(self,points):
        scale=.001
        src=Rotation.from_quat(self.pose[3:]).inv().apply(
            np.asarray(points)-self.pose[:3]-self.offset)/scale
        return -self.grid.sample(src)*scale-.000665


def invoke(fn,grid,pose,offset,q,ids,lengths):
    dims=np.array([grid.values.shape[2],grid.values.shape[1],grid.values.shape[0]],np.int32)
    state=np.ascontiguousarray(q,np.float64)
    nodes=np.ascontiguousarray(ids,np.int32)
    length=np.ascontiguousarray(lengths,np.float64)
    field=np.ascontiguousarray(grid.values,np.float32)
    origin=np.ascontiguousarray(grid.origin,np.float64)
    spacing=np.ascontiguousarray(grid.spacing,np.float64)
    pose=np.ascontiguousarray(pose,np.float64)
    offset=np.ascontiguousarray(offset,np.float64)
    output=np.zeros(7,np.float64)
    ptr=lambda a,t:a.ctypes.data_as(ctypes.POINTER(t))
    rc=fn(ptr(state,ctypes.c_double),len(state),ptr(nodes,ctypes.c_int32),
          ptr(length,ctypes.c_double),len(length),ptr(field,ctypes.c_float),
          ptr(dims,ctypes.c_int32),ptr(origin,ctypes.c_double),ptr(spacing,ctypes.c_double),
          ptr(pose,ctypes.c_double),ptr(offset,ctypes.c_double),
          .001,.000665,.0001,.00025,2.,.00005,ptr(output,ctypes.c_double))
    assert rc==0,rc
    return output


def main():
    grid=Grid()
    with tempfile.TemporaryDirectory() as tmp:
        lib=pathlib.Path(tmp)/"libnative_guard_test.so"
        subprocess.run(["g++","-std=c++17","-O2","-fPIC","-shared",
                        str(HERE/"native/FastBeamSDF.cpp"),"-o",str(lib)],
                       check=True)
        fn=ctypes.CDLL(str(lib)).mcr_fast_coarse_sdf_v1
        fn.restype=ctypes.c_int
        fn.argtypes=[
            ctypes.POINTER(ctypes.c_double),ctypes.c_int32,
            ctypes.POINTER(ctypes.c_int32),ctypes.POINTER(ctypes.c_double),ctypes.c_int32,
            ctypes.POINTER(ctypes.c_float),ctypes.POINTER(ctypes.c_int32),
            ctypes.POINTER(ctypes.c_double),ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),ctypes.POINTER(ctypes.c_double),
            ctypes.c_double,ctypes.c_double,ctypes.c_double,
            ctypes.c_double,ctypes.c_double,ctypes.c_double,
            ctypes.POINTER(ctypes.c_double)]
        cases=0
        for angle in (0.,.4,-.8):
            quat=Rotation.from_euler('z',angle).as_quat()
            pose=np.array([.037,.020,.008,*quat])
            offset=np.array([.002,-.001,.003])
            for y in (0.,3.4,3.9,4.3,5.0,20.):
                for curved in (False,True):
                    local=np.array([[-6,y,0], [0,y+(1.0 if curved else 0),0],[6,y,0.]])*0.001
                    sim=Rotation.from_quat(quat).apply(local)+pose[:3]+offset
                    direction=Rotation.from_quat(quat) * Rotation.from_euler('z',.25 if curved else 0)
                    q=np.column_stack([sim,np.tile(direction.as_quat(),(3,1))])
                    ids=np.array([[0,1],[1,2]],np.int32)
                    lengths=np.array([.006,.006])
                    adapter=Adapter(grid,pose,offset,lengths,ids)
                    python=coarse_beam_clearance_guard(adapter,q)
                    native=invoke(fn,grid,pose,offset,q,ids,lengths)
                    assert bool(native[6])==bool(python["safe"]),(y,curved,native,python)
                    if bool(native[6]):
                        assert not bool(native[4])
                        assert abs(native[0]-python["minimum_clearance_m"])<2e-7
                        assert abs(native[1]-python["lower_bound_m"])<2e-7
                    cases+=1
        print("NATIVE_COARSE_GUARD_PARITY_PASS",cases,"synthetic cases")


if __name__=="__main__":
    main()
