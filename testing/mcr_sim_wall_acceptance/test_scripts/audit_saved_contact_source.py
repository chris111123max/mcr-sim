"""Read installed SOFA source and saved runtime facts; never initialize SOFA."""
import sys,json,hashlib,re
from pathlib import Path
from collections import Counter

ROOT=Path('/data/home/3220251075/mcr_sim/mcr_env/sofa/src/sofa/modules/SofaConstraint/src/SofaConstraint')
REPO=Path(__file__).resolve().parents[4]

def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def evidence(p,lo,hi):
    lines=p.read_text().splitlines()
    return {'path':str(p),'sha256':digest(p),'start':lo,'end':hi,'source':'\n'.join('%d: %s'%(i+1,lines[i]) for i in range(lo-1,min(hi,len(lines))))}

def main(source,out):
    source=Path(source);out=Path(out);report=json.loads((source/'report.json').read_text());protected=json.loads((source/'protected_sha256.json').read_text())
    required=['mcr_simulator.py','mcr_environment.py','mcr_instrument.py','sofa_simulation.py','config.py']
    for n in required:
        p=REPO/'python/mcr_sim'/n
        if protected.get(str(p))!=digest(p):raise ValueError('Production differs from saved run: '+n)
    f=ROOT/'FrictionContact.inl';l=ROOT/'LCPConstraintSolver.cpp';u=ROOT/'UnilateralInteractionConstraint.inl'
    assert 'initData(&mu, 0.8' in f.read_text()
    assert 'W33[c1].New_GS_State(_mu' in l.read_text()
    assert 'residuals->push_back(error);' in l.read_text()
    assert 'getConstraintResolution' not in l.read_text()
    rows=[json.loads(x) for x in (source/'physical_substeps.jsonl').read_text().splitlines()]
    object_rows=[r for r in rows if r['observed_contact_constraint_objects']]
    for r in object_rows:
        assert {o['class'] for o in r['observed_contact_constraint_objects']}=={'FrictionContact','UnilateralInteractionConstraint'}
        assert all(o['path']=='/mag_instrument/CollisionModel/TriangleCollisionModel-PointCollisionModel' for o in r['observed_contact_constraint_objects'])
        assert r['solver']['lcp']['data']['build_lcp']==0 and r['solver']['lcp']['data']['mu']==.1 and r['solver']['lcp']['data']['displayDebug']==0
    raw_mu=sorted({o['data']['mu'] for r in object_rows for o in r['observed_contact_constraint_objects'] if o['class']=='FrictionContact'})
    sweeps=json.loads((out/'lcp_sweeps.json').read_text());lookup={r['physical_substep']:r['sweeps_from_residual_vector'] for r in sweeps}
    result={'status':'PASS_SOURCE_AND_SAVED_PAIR_ATTRIBUTION','new_sofa_integrations':0,'production_matches_saved_run':True,'contact_pair':{'status':'CONFIRMED_BY_SAVED_OBJECTS_AND_HASH_MATCHED_UNIQUE_SCENE','vessel_triangle':'/CollisionModel/TriangleCollisionModel','catheter_point':'/mag_instrument/CollisionModel/PointCollisionModel','response_object':object_rows[0]['observed_contact_constraint_objects'][0]['path'],'first_object_substep':object_rows[0]['physical_substep'],'object_present_substeps':len(object_rows),'evidence':'Only vessel constructs TriangleCollisionModel; vessel PointCollisionModel disabled, and only catheter constructs PointCollisionModel in this scene. Recorded response name identifies Triangle-Point; response path is catheter collision node, not a second vessel. Source activateMappers constructs constraint using actual collision-model mapped states.','direct_runtime_model_pointer_getter':'NOT_OBSERVED; not captured/exposed in saved Python snapshots','other_pair_types':'NOT_VERIFIED; no evidence extrapolated to Triangle-Line or self collision'},'friction':{'FrictionContact_mu_observed':raw_mu,'origin':'FrictionContact.inl constructor default0.8; production CollisionResponse sets response type but no responseParams override.','role_of_0_8':'Passed into addContact and stored as c.mu; positive coefficient selects three normal/tangent constraint rows. Generic constraint-resolution path would use c.mu, but current LCP solver does not call getConstraintResolution.','LCP_mu_observed':.1,'origin_of_0_1':'config.py SofaConfig.friction=0.1 passed by Simulator into LCPConstraintSolver.mu.','effective_unbuilt_GS_friction':.1,'proof':'Saved build_lcp=0, displayDebug=0. solveSystem calls gaussseidel_unbuilt; build_problem_info and nlcp_gaussseidel_unbuilt read solver mu; W33.New_GS_State receives that same0.1 for stick/slip update. Therefore coefficients do not multiply or average.'},'iteration_measurement':{'status':'SOURCE_DERIVED_OBSERVED','direct_getter':'NOT_OBSERVED','method':'Error vector cleared before unbuilt solve; one residual push per actual GS sweep. Count saved numeric entries. displayDebug=0 rules out a second debug solve.','whole_run_histogram':dict(Counter(x['sweeps_from_residual_vector'] for x in sweeps)),'contact_frame_histogram':dict(Counter(lookup[r['physical_substep']] for r in object_rows)),'warning':'0 empty-vector sweeps does not by itself establish contact count0.'},'contact_count':'NOT_OBSERVED','contact_force':'NOT_OBSERVED','evidence':[evidence(ROOT.parents[3]/'SofaKernel/modules/SofaHelper/src/sofa/helper/LCPcalc.cpp',597,651),evidence(f,42,58),evidence(f,130,151),evidence(f,203,240),evidence(l,166,171),evidence(l,715,724),evidence(l,841,875),evidence(l,1026,1062),evidence(l,1093,1118),evidence(u,88,96),evidence(u,316,341),evidence(u,361,372),evidence(REPO/'python/mcr_sim/config.py',74,81),evidence(REPO/'python/mcr_sim/mcr_simulator.py',15,26),evidence(REPO/'python/mcr_sim/sofa_simulation.py',118,126),evidence(REPO/'python/mcr_sim/mcr_environment.py',103,150),evidence(REPO/'python/mcr_sim/mcr_instrument.py',62,68)],'limits':['Attribution uses saved objects plus unchanged scene code, not a newly observed model pointer.','Installed C++ source is inspected and hashed; no recompilation or solver parameter change.']}
    (out/'source_contact_audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='evidence'},indent=2))
if __name__=='__main__':main(sys.argv[1],sys.argv[2])
