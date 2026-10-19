from pathlib import Path
import json, hashlib
p=Path('testing/beam_feasible_domain/native_repeat_solve_consistency')
old=p.parent/'jacobian_world_frame_validation'
(p/'native').mkdir(parents=True,exist_ok=True)
(p/'results').mkdir(exist_ok=True)
cases=json.loads((old/'cases.json').read_text())
cases=[c for c in cases if c['worker'] in (9,6,23)]
(p/'cases.json').write_text(json.dumps(cases,indent=2))
for f in ('world_rows.py','preflight.py'):
    (p/f).write_text((old/f).read_text())
cm=(old/'native/CMakeLists.txt').read_text().replace('MCRWorldFrameValidation','MCRRepeatSolveConsistency')
(p/'native/CMakeLists.txt').write_text(cm)
s=(old/'native/TestOnlyRelinearizingGenericConstraintSolver.cpp').read_text()
s=s.replace('#include <algorithm>', '#include <fstream>\n#include <iomanip>\n#include <sofa/core/behavior/MechanicalState.h>\n#include <sofa/defaulttype/RigidTypes.h>\n#include <sofa/defaulttype/VecTypes.h>\n#include <algorithm>')
s=s.replace(': d_enabled(', ': d_auditPath(initData(&d_auditPath, std::string(""), "auditPath", "Read-only native audit JSONL path"))\n        , d_enabled(')
s=s.replace('"MCRWorldFrameValidation"','"MCRRepeatSolveConsistency"')
instrument=r'''
    using State=sofa::core::behavior::MechanicalState<sofa::defaulttype::Rigid3Types>;
    template<class V> void vec(std::ostream& o,const V& v,int n) {
        o << "["; for(size_t i=0;i<v.size();++i) {if(i)o<<",";o<<"[";
        for(int j=0;j<n;++j){if(j)o<<",";o<<v[i][j];}o<<"]";}o<<"]";
    }
    void audit(const char* phase,const ConstraintParams* p) {
        if(!d_enabled.getValue()||d_auditPath.getValue().empty())return;
        std::vector<State*> rigidStates;getContext()->getObjects(&rigidStates,sofa::core::objectmodel::BaseContext::SearchDown);
        State* s=nullptr;for(auto* candidate:rigidStates)if(candidate->getName()=="DOFs")s=candidate;
        std::ofstream o(d_auditPath.getValue(),std::ios::app);o<<std::setprecision(17);
        o<<"{\"stage\":"<<(d_nativeSolveCalls.getValue()+(std::string(phase).find("map")!=std::string::npos?0:1))<<",\"phase\":\""<<phase<<"\"";
        if(s){
            o<<",\"beam_path\":\""<<s->getPathName()<<"\"";
            auto* xp=p->readX(s);auto* vp=p->readV(s);
            o<<",\"reference_aliases_position\":"<<(xp==s->read(sofa::core::ConstVecCoordId::position())?"true":"false");
            o<<",\"reference_aliases_free\":"<<(xp==s->read(sofa::core::ConstVecCoordId::freePosition())?"true":"false");
            o<<",\"position\":";vec(o,s->read(sofa::core::ConstVecCoordId::position())->getValue(),7);
            o<<",\"free_position\":";vec(o,s->read(sofa::core::ConstVecCoordId::freePosition())->getValue(),7);
            o<<",\"reference_x\":";vec(o,xp->getValue(),7);
            o<<",\"velocity\":";vec(o,s->read(sofa::core::ConstVecDerivId::velocity())->getValue(),6);
            o<<",\"free_velocity\":";vec(o,s->read(sofa::core::ConstVecDerivId::freeVelocity())->getValue(),6);
            o<<",\"reference_v\":";vec(o,vp->getValue(),6);
            o<<",\"native_dx\":";vec(o,s->read(getDx().getId(s))->getValue(),6);
            o<<",\"applied_dx\":";vec(o,s->read(p->dx().getId(s))->getValue(),6);
            o<<",\"jacobian\":[";bool first=true;const auto& J=p->readJ(s)->getValue();
            for(auto ri=J.begin();ri!=J.end();++ri)for(auto ci=ri.begin();ci!=ri.end();++ci){
                if(!first)o<<",";first=false;o<<"["<<ri.index()<<","<<ci.index();
                for(int j=0;j<6;++j)o<<","<<ci.val()[j];o<<"]";
            }o<<"]";
        }
        using VS=sofa::core::behavior::MechanicalState<sofa::defaulttype::Vec3Types>;
        std::vector<VS*> states;getContext()->getObjects(&states,sofa::core::objectmodel::BaseContext::SearchDown);
        o<<",\"mapped\":[";bool firstState=true;
        for(auto* cs:states)if(cs->getName()=="CollisionDOFs"){
            if(!firstState)o<<",";firstState=false;o<<"{\"path\":\""<<cs->getPathName()<<"\",\"position\":";
            vec(o,cs->read(sofa::core::ConstVecCoordId::position())->getValue(),3);
            o<<",\"free_position\":";vec(o,cs->read(sofa::core::ConstVecCoordId::freePosition())->getValue(),3);
            o<<",\"reference_x\":";vec(o,p->readX(cs)->getValue(),3);o<<"}";
        }o<<"]";
        if(current_cp){int n=current_cp->getDimension();
            o<<",\"dimension\":"<<n<<",\"rhs\":[";
            for(int i=0;i<n;++i){if(i)o<<",";o<<current_cp->dFree[i];}o<<"],\"lambda\":[";
            for(int i=0;i<n;++i){if(i)o<<",";o<<current_cp->f[i];}o<<"],\"W\":[";
            for(int i=0;i<n;++i){if(i)o<<",";o<<"[";for(int j=0;j<n;++j){if(j)o<<",";o<<current_cp->getW()[i][j];}o<<"]";}o<<"]";
            o<<",\"error\":"<<current_cp->currentError<<",\"iterations\":"<<current_cp->currentIterations;
        }o<<"}\n";
    }
    bool prepareStates(const ConstraintParams*p,sofa::core::MultiVecId x,sofa::core::MultiVecId v) override {
        bool r=GenericSolver::prepareStates(p,x,v);audit("prepared",p);return r;
    }
    bool buildSystem(const ConstraintParams*p,sofa::core::MultiVecId x,sofa::core::MultiVecId v) override {
        bool r=GenericSolver::buildSystem(p,x,v);audit("built",p);return r;
    }
    bool solveSystem(const ConstraintParams*p,sofa::core::MultiVecId x,sofa::core::MultiVecId v) override {
        bool r=GenericSolver::solveSystem(p,x,v);audit("solved",p);return r;
    }
    bool applyCorrection(const ConstraintParams*p,sofa::core::MultiVecId x,sofa::core::MultiVecId v) override {
        audit("pre_apply",p);bool r=GenericSolver::applyCorrection(p,x,v);audit("post_apply",p);return r;
    }
'''
s=s.replace('\nprivate:\n', '\n'+instrument+'\nprivate:\n    sofa::core::objectmodel::Data<std::string> d_auditPath;\n')
s=s.replace('propagate.execute(getContext());','audit("pre_map", &stageParams);\n                propagate.execute(getContext());\n                audit("post_map", &stageParams);')
(p/'native/TestOnlyRelinearizingGenericConstraintSolver.cpp').write_text(s)
r=(old/'run_case.py').read_text()
r=r.replace("[('A1',1),('A2',2),('B1',1)]", "([('P1',1),('P2',2)] if CASE['worker']==9 else [('P1',1)])")
r=r.replace("self.branch='B2'; self.passes=2", "self.branch=('P3' if CASE['worker']==9 else 'P2'); self.passes=(3 if CASE['worker']==9 else 2)")
r=r.replace("WORLD_MODE=self.branch.startswith('B')", "WORLD_MODE=True")
r=r.replace("self.passes in (1,2)","self.passes in (1,2,3)")
r=r.replace("self.native.requestedPasses.value=self.passes", "self.native.auditPath.value=str(OUT/f'{self.branch}_sub{self._beam_substep}_native.jsonl')\n            self.native.requestedPasses.value=self.passes")
r=r.replace("'world' if self.branch.startswith('B') else 'local'", "'world'")
r=r.replace('libMCRWorldFrameValidation.so','libMCRRepeatSolveConsistency.so')
(p/'run_case.py').write_text(r)
sources=['mcr_sim/beam_safety_env.py','mcr_sim/beam_sdf_unilateral_fast.py','mcr_sim/native/beam_unilateral/BeamLinearizedUnilateralConstraint.cpp','training/py/train_ppo.py','training/py/train_ppo_beam_unilateral.py']
(p/'results/production_before.json').write_text(json.dumps({f:hashlib.sha256(Path(f).read_bytes()).hexdigest() for f in sources},indent=2))
print('SETUP_COMPLETE',[(c['id'],c['worker']) for c in cases])
