"""Production exact nodal-load adapter for SOFA21 ConstantForceField updates."""
import numpy as np
class ForceData:
 def __init__(self,owner):self.owner=owner
 @property
 def value(self):return [np.asarray(f.forces.value).reshape(-1,6)[0].tolist() for f in self.owner.fields]
 @value.setter
 def value(self,values):
  a=np.asarray(values,dtype=float).reshape(len(self.owner.fields),6)
  self.owner.requested_forces=a.copy()
  for f,v in zip(self.owner.fields,a):f.forces.value=[v.tolist()]
class IndexData:
 def __init__(self,owner):self.owner=owner
 @property
 def value(self):return [int(np.asarray(f.indices.value).reshape(-1)[0]) for f in self.owner.fields]
class NodalForceProxy:
 def __init__(self,fields):
  self.fields=fields;self.requested_forces=np.zeros((len(fields),6));self.forces=ForceData(self);self.indices=IndexData(self)
