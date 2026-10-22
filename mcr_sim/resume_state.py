"""Fail-closed validation and chronological replay migration for trusted checkpoints."""
import hashlib,random
from pathlib import Path
import numpy as np
import torch
from .serialization import write_json

def sha256(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def finite_tree(x):
    if torch.is_tensor(x):
        if not bool(torch.isfinite(x).all()):raise ValueError('Nonfinite checkpoint tensor')
    elif isinstance(x,np.ndarray) and np.issubdtype(x.dtype,np.number):
        if not np.isfinite(x).all():raise ValueError('Nonfinite checkpoint array')
    elif isinstance(x,dict):
        for v in x.values():finite_tree(v)
    elif isinstance(x,(list,tuple)):
        for v in x:finite_tree(v)

def restore(path,agent,replay,output=None):
    c=torch.load(path,map_location='cpu')
    required=['step','networks','target','log_alpha','actor_optimizer','critic_optimizer','alpha_optimizer','python_rng','numpy_rng','torch_rng','npu_rng','replay','replay_episodes']
    if any(k not in c for k in required):raise ValueError('Incomplete checkpoint: missing required training state')
    if c.get('replay_format_version')!=replay.FORMAT_VERSION:raise ValueError('Incompatible replay format')
    if not 0<len(c['replay'])<=replay.data.maxlen:raise ValueError('Missing/oversized saved replay')
    for key in ['networks','target','log_alpha','actor_optimizer','critic_optimizer','alpha_optimizer']:
        finite_tree(c[key])
    for key in ['actor_optimizer','critic_optimizer','alpha_optimizer']:
        if not c[key]['state']:raise ValueError('Missing optimizer moments')
    previous={}
    for t in c['replay']:
        replay._validate_observation(t.observation);replay._validate_observation(t.next_observation)
        if not np.isfinite(t.action).all() or t.action.shape!=(3,) or not np.isfinite(t.reward):raise ValueError('Invalid replay record')
        replay._sequence(0,[t])
        expected=np.concatenate((t.observation_history,t.next_observation[None]),axis=0)[-replay.sequence_length:]
        if not np.array_equal(expected,t.next_observation_history):raise ValueError('Invalid next history')
        old=previous.get(t.env_id)
        if old is not None:
            if t.episode<old.episode:raise ValueError('Replay chronology reversed')
            if t.episode==old.episode:
                if old.terminated or old.truncated or not np.array_equal(old.next_observation,t.observation):raise ValueError('Replay episode discontinuity')
                if not np.array_equal(old.next_observation_history,t.observation_history):raise ValueError('Replay history discontinuity')
        previous[t.env_id]=t
    agent.restore(c)
    replay.data.extend(c['replay'])
    replay.episodes={int(k):int(v)+1 for k,v in c['replay_episodes'].items()}
    random.setstate(c['python_rng']);np.random.set_state(c['numpy_rng']);torch.set_rng_state(c['torch_rng'].cpu())
    if agent.device.type=='npu':torch.npu.set_rng_state(c['npu_rng'].cpu(),agent.device)
    raw_updates=c.get('updates')
    updates=None if raw_updates is None else int(raw_updates)
    agent.restored_training_state=dict(c.get('training_state',{}))
    report=dict(source=str(Path(path).resolve()),sha256=sha256(path),step=int(c['step']),
                updates=updates,update_count_source='saved' if updates is not None else 'unknown',
                replay_capacity=replay.data.maxlen,replay_records=len(replay),
                saved_replay_records=len(c['replay']),history_frames=8,observation_dim=41,action_dim=3,
                model_target_optimizers_restored=True,rng_restored=True,physical_episodes_resumed=False,
                environment_reset_note='Fresh physical episodes; replay histories preserved; online histories reset.')
    agent.resume_metadata=report
    start=int(c['step'])
    # Keep only ring-owned transitions and actual optimizer tensors, never the
    # outer checkpoint/replay list or per-environment validation references.
    del c,previous,old,t,expected
    if output:write_json(Path(output)/'resume.json',report)
    return start,updates
