"""Read cgroup v1/v2 and process memory without changing system limits."""
import os,time,json
from pathlib import Path
GIB=1024**3

def snapshot(pids=()):
    v1=Path('/sys/fs/cgroup/memory')
    base=v1 if (v1/'memory.limit_in_bytes').exists() else Path('/sys/fs/cgroup')
    names={'current':'memory.usage_in_bytes','peak':'memory.max_usage_in_bytes','limit':'memory.limit_in_bytes','events':'memory.failcnt'} if base==v1 else {'current':'memory.current','peak':'memory.peak','limit':'memory.max','events':'memory.events'}
    result={'time':time.time(),'processes':{}}
    for k,n in names.items():
        try:
            t=(base/n).read_text().strip();result[k]=int(t) if t.isdigit() else t
        except OSError:result[k]=None
    for pid in set([os.getpid()]+list(pids)):
        try:
            vals={}
            for line in Path('/proc/%d/smaps_rollup'%pid).read_text().splitlines():
                if line.startswith(('Rss:','Pss:')):
                    k,v,*_=line.split();vals[k[:-1].lower()]=int(v)*1024
            result['processes'][str(pid)]=vals
        except OSError:result['processes'][str(pid)]={'exited':True}
    return result

def safe(s,reserve_gib=24):
    return isinstance(s.get('limit'),int) and isinstance(s.get('current'),int) and s['limit']-s['current']>reserve_gib*GIB

def record(path,event,step=0,envs=(),**extra):
    s=snapshot([e.process.pid for e in envs]);s.update(event=event,step=step,num_envs=len(envs),**extra)
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('a') as f:f.write(json.dumps(s)+'\n')
    return s
