from pathlib import Path
import json,hashlib,os
H=Path(__file__).resolve().parent;D=H/'research_sources'
commit='d300cc9af550afbd5c2167470676a5a7d47381a1'
tree={i['path']:i['sha'] for i in json.loads((D/'sofa_tree.json').read_text())['tree']}
path='SofaKernel/modules/SofaCore/src/sofa/core/behavior/PairInteractionConstraint.inl'
b=(Path(os.environ['SOFA_SRC'])/path).read_bytes()
blob=hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest();assert blob==tree[path]
filename='sofa__'+path.replace('/','__');(D/filename).write_bytes(b)
manifest=json.loads((D/'manifest.json').read_text())
manifest=[x for x in manifest if x['path']!=path]
manifest.append(dict(repo='sofa-framework/sofa',commit=commit,path=path,url=f'https://github.com/sofa-framework/sofa/blob/{commit}/{path}',file=filename,
    sha256=hashlib.sha256(b).hexdigest(),github_blob_sha=blob,installed_equal=True,transport='github_tree_attested_installed_bytes',status='OK'))
(D/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
audit='''# Current implementation audit (read-only)

The existing tests in native_repeat_solve_consistency, jacobian_world_frame_validation, multistate_native_safety and the production Beam row builder/environment are reused, not modified.

1. Normal FreeMotionAnimationLoop sets ConstraintParams X/V to freePosition/freeVelocity, detects contacts once and invokes the native Generic solver once.
2. Native UnilateralInteractionConstraint::getPositionViolation reads contact mechanical freePosition. Generic builds dFree/W/J, solves f and applies native correction from the selected reference X/V.
3. The experimental repeat wrapper changes X/V to committed position/velocity after solve1, propagates those vectors with the native mechanical visitor, rebuilds world-frame Beam rows on that committed state, and calls Generic again without advancing physical time.
4. Original contact callback ignores PairInteractionConstraint's passed X coordinates: freePosition stays unchanged, so dFree is reproduced. Correction nevertheless uses the newly selected committed position; the full old contact response is added again. Actual prior dx clearing/lambda/mapping data already rule out stale dx.
5. The isolated extension installs a derived native contact through virtual FrictionContact::activateMappers. Original activateMappers then performs all remaining ordinary initialization/mapping/contact work. A/B call the original violation method; C calculates the original formula on native readX-selected coordinates. The derived solver only scopes this switch around native buildSystem.
6. Contact normal/tangent bases and collision manifold are held fixed within this substep, as in the prior test. Their local Jacobian is inherited unchanged; the native buildSystem resets, rebuilds and accumulates the parent Jacobian after native propagation. Beam rows are rebuilt at the same reference state. Audits read real dFree/f/W, parent and mapped J, selected coordinates, actual dx, contact points/bases and committed geometry.
7. This is not a production fix, a new collision detector, a continuous nonpenetration guarantee, or a proof of full nonlinear dynamic equilibrium. A frozen contact manifold remains a scope limitation. No original free or committed coordinates are directly written in a real frame. Native correction is the only state-changing correction.
8. Synthetic unit initialization writes only newly constructed standalone mechanical objects before any simulation and never receives a saved/real state. A first fixture allocation failure was corrected before any real replay. Runtime shared libraries and production source are untouched. Eigen 3.3.7 headers, needed by SOFA contact mapper headers and absent from the current filesystem, are downloaded only into this test results/build_deps; they are not installed into any environment.

The original production tolerance is preserved through each saved action prefix; the matched target uses world angular Jacobian and the previously audited 1e-9 tolerance for every A/B/C branch. Requested margin, dense selection, FD steps, dt and physics substep count do not change.
'''
(H/'implementation_audit.md').write_text(audit)
print('PAIR_INTERFACE_GITHUB_BLOB_VERIFIED',blob)
