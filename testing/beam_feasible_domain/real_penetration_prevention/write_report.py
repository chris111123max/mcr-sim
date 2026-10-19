from pathlib import Path
import json
H=Path(__file__).resolve().parent;R=H/'results'
a=json.loads((R/'analysis.json').read_text());run=a['run']
records=[]
for p in sorted((H.parent/'formal_training_failure').glob('worker_*/most_dangerous_metadata.json')):
 m=json.loads(p.read_text());records.append({'metadata':str(p.resolve()),'worker':m['worker_env'],'step':m['rl_step'],'substep':m['substep'],'q_free_clearance_mm':m['q_free_clearance_mm'],'q_committed_clearance_mm':m['q_committed_clearance_mm']})
(R/'existing_snapshot_audit.json').write_text(json.dumps(records,indent=2))
assert min(x['q_free_clearance_mm'] for x in records)>0
lines=['# REAL PENETRATION PREVENTION TEST','',
 '- PRODUCTION MODIFIED = NO (5 protected-file SHA256 hashes unchanged)',
 '- PPO TRAINING STARTED = NO','- REAL UNSAFE FREE STATES = 0','- VALID SAME-STATE REPLAYS = 0',
 '- TIP PENETRATION CASES = 0','- BODY PENETRATION CASES = 0','- EXTRA PHYSICS SUBSTEPS = 0','']
for b,name in [('A','Original production Jacobian, single native solve'),('B','World Jacobian, single native solve'),('C','World Jacobian + consistent reference, max two solves per original substep')]:
 lines += ['## '+b+': '+name,'','- Real penetration prevented = 0/0 (no unsafe samples; not 100% success)',
 '- Worst committed clearance = N/A: unsafe branch not executed','- Body penetration count = N/A: unsafe branch not executed','']
lines += ['- CONTACT RHS CONSISTENCY = unit PASS; unsafe-state test INCONCLUSIVE',
 '- NATIVE EXECUTION AUDIT = INCONCLUSIVE',
 '- ENTIRE BEAM DENSE CHECK = INCONCLUSIVE: no unsafe target available',
 '- TWO-SUBSTEP NONPENETRATION = INCONCLUSIVE','- WORST FAILURE STATE = N/A',
 '- HISTORICAL -0.766410 MM ROOT CAUSE = UNPROVEN','- FINAL RESULT = INCONCLUSIVE','',
 '## Actual execution','',
 'All new files and execution were on the remote server, under this testing directory. No local staging files used.',
 'Latest verified reference-consistency native binary reused. Old inconsistent Contact RHS repetition was not used.',
 'Native callback unit actually ran: original/free RHS -0.100 mm; updated/current RHS +0.050 mm; baseline exact equality PASS. This is a synthetic unit test, not a catheter penetration case.',
 f'Existing {len(records)} most-dangerous snapshots all have positive free clearance; minimum +{min(x["q_free_clearance_mm"] for x in records):.9f} mm. They capture committed risk, not deepest free risk.',
 f'One bounded B02/target_04 saved-action prefix, worker 23 seed 65: {run["rl_steps_executed"]} RL steps / {2*run["rl_steps_executed"]} original physics substeps, {run["wall_s"]:.3f} s.',
 f'Actual CollisionBeginEvent production-dense free minimum = +{run["worst_free_clearance_mm"]:.9f} mm; negative free substeps = {run["free_negative_substeps"]}; captured states = {run["captured_states"]}.',
 'Capture threshold -0.050 mm, but not even a negative free clearance occurred. Consequently A/B/C unsafe branches were not run. No fabricated lambda/RHS/committed prevention measurements.',
 'The action source is the formal epoch005 snapshot, not historical epoch74/seed15204. Different experimental versions were not mixed.',
 'No second prefix, seed search, full-episode sweep, or PPO training. Historical negative free trace was not restored with full physical context and is not counted as a valid same-state replay.','',
 '## Unverified implementation','',
 'run_penetration.py implements live process fork, original two-substep continuity and independent whole deployed-Beam radius-subtracted SDF checking.',
 'Independent checker uses derivative-bounded spacing <=10 micrometres, rejects nonfinite SDF, reports invalid normals and all negative samples. Inactive tail outside inlet is explicitly excluded. Tip classification uses last three active elements, matching read production defaults.',
 'These unsafe branches and independent checker were not exercised because there were no captured states. Their correctness is NOT claimed verified.',
 'The diagnostic guard may record penetration and allow only the already scheduled second substep; it does not rewrite positions or change safety criteria. This path was not exercised.',
 'No production, dt, safety math, PPO, reward, observation, action or done modification.','',
 '## Answer','',
 'INCONCLUSIVE: this run cannot determine whether SOFA prevents real free-state wall penetration. Positive free states cannot substitute for the requested unsafe-state acceptance.',
 'No evidence to assign a failure to detection, row construction, solver, or committed update. Historical -0.766410 mm root cause remains UNPROVEN.',
 'Remaining requirement: a restorable real unsafe CollisionBegin state, then actual A/B/C native replay. Stopped after the one bounded prefix; no automatic expansion.']
(R/'final_report.md').write_text('\n'.join(lines)+'\n')
print(R/'final_report.md')
