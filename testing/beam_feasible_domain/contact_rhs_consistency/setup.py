"""Generate isolated test extension from inspected, preserved prior test sources."""
from pathlib import Path
import hashlib, json, os
HERE=Path(__file__).resolve().parent
OLD=HERE.parent/'native_repeat_solve_consistency'
SRC=Path(os.environ['SOFA_SRC'])
(HERE/'native').mkdir(exist_ok=True)
(HERE/'results').mkdir(exist_ok=True)
for name in ('cases.json','world_rows.py','preflight.py'):
    (HERE/name).write_bytes((OLD/name).read_bytes())
prod=HERE.parents[2]
files=['mcr_sim/beam_safety_env.py','mcr_sim/beam_sdf_unilateral_fast.py','mcr_sim/native/beam_unilateral/BeamLinearizedUnilateralConstraint.cpp','training/py/train_ppo.py','training/py/train_ppo_beam_unilateral.py']
(HERE/'results/production_before.json').write_text(json.dumps({f:hashlib.sha256((prod/f).read_bytes()).hexdigest() for f in files},indent=2))
original=(SRC/'modules/SofaConstraint/src/SofaConstraint/UnilateralInteractionConstraint.inl').read_text()
body=original.split('void UnilateralInteractionConstraint<DataTypes>::getPositionViolation(linearalgebra::BaseVector *v)\n{',1)[1].split('\n}\n',1)[0]
body=body[body.index('\n    Real dfree'):]
header='''// Adapted small method from official SOFA v21.12.00, commit d300cc9af550afbd5c2167470676a5a7d47381a1.
// Copyright (c) 2006 INRIA, USTL, UJF, CNRS, MGH. LGPL-2.1-or-later.
// https://github.com/sofa-framework/sofa/blob/d300cc9af550afbd5c2167470676a5a7d47381a1/modules/SofaConstraint/src/SofaConstraint/UnilateralInteractionConstraint.inl
#pragma once
#include <SofaConstraint/UnilateralInteractionConstraint.inl>
#include <SofaConstraint/FrictionContact.inl>
#include <SofaMeshCollision/RigidContactMapper.inl>
#include <SofaMeshCollision/BarycentricContactMapper.inl>
#include <SofaBaseMechanics/MechanicalObject.h>
#include <sofa/linearalgebra/FullVector.h>
#include <sstream>
namespace mcr::contact_reference_test {
namespace helper=sofa::helper;
inline thread_local bool consistentReference = false;
using NativeContact = sofa::component::constraintset::UnilateralInteractionConstraint<sofa::defaulttype::Vec3Types>;
class ReferenceUnilateralContact final : public NativeContact {
public:
    SOFA_CLASS(ReferenceUnilateralContact, NativeContact);
    ReferenceUnilateralContact(MechanicalState* a=nullptr,MechanicalState* b=nullptr):NativeContact(a,b){}
    const sofa::core::ConstraintParams* reference=nullptr;
    std::string lastGeometry="[]";
    void getConstraintViolation(const sofa::core::ConstraintParams*p,sofa::linearalgebra::BaseVector*v,
      const DataVecCoord&x1,const DataVecCoord&x2,const DataVecDeriv&v1,const DataVecDeriv&v2) override {
        reference=p;
        NativeContact::getConstraintViolation(p,v,x1,x2,v1,v2);
        reference=nullptr;
    }
    void evaluateSelected(sofa::linearalgebra::BaseVector*v,const VecCoord&QfreeVec,const VecCoord&PfreeVec){
'''+body+'''
    }
    void getPositionViolation(sofa::linearalgebra::BaseVector*v) override {
        const bool selected=consistentReference && reference;
        const auto& Q=selected?reference->readX(getMState1())->getValue():getMState1()->read(sofa::core::ConstVecCoordId::freePosition())->getValue();
        const auto& P=selected?reference->readX(getMState2())->getValue():getMState2()->read(sofa::core::ConstVecCoordId::freePosition())->getValue();
        if(selected)evaluateSelected(v,Q,P);else NativeContact::getPositionViolation(v);
        std::ostringstream o;o<<std::setprecision(17)<<"[";bool first=true;
        for(const auto&c:contacts){if(!first)o<<",";first=false;
            o<<"{\\"row\\":"<<c.id<<",\\"mu\\":"<<c.mu<<",\\"distance\\":"<<c.contactDistance
             <<",\\"selected_reference\\":"<<(selected?"true":"false")<<",\\"normal_gap\\":"<<dot(P[c.m2]-Q[c.m1],c.norm)-c.contactDistance
             <<",\\"m1\\":"<<c.m1<<",\\"m2\\":"<<c.m2<<",\\"state1\\":\\""<<getMState1()->getPathName()<<"\\",\\"state2\\":\\""<<getMState2()->getPathName()<<"\\"";
            auto write=[&](const char*name,const auto&a){o<<",\\""<<name<<"\\":["<<a[0]<<","<<a[1]<<","<<a[2]<<"]";};
            write("P",P[c.m2]);write("Q",Q[c.m1]);write("normal",c.norm);write("tangent",c.t);write("second_tangent",c.s);
            write("anchor_P",c.P);write("anchor_Q",c.Q);
            const auto& PX=reference->readX(getMState2())->getValue();const auto& QX=reference->readX(getMState1())->getValue();
            write("cparams_P",PX[c.m2]);write("cparams_Q",QX[c.m1]);
            o<<",\\"rhs_normal\\":"<<v->element(c.id)<<",\\"rhs_tangent\\":"<<(c.mu>0?v->element(c.id+1):0)
             <<",\\"rhs_second_tangent\\":"<<(c.mu>0?v->element(c.id+2):0)<<"}";
        }o<<"]";lastGeometry=o.str();
    }
    void setUnitRow(){contacts[0].id=0;}
};
template<class M1,class M2> class ReferenceFrictionContact:public sofa::component::collision::FrictionContact<M1,M2>{
    using Base=sofa::component::collision::FrictionContact<M1,M2>;
public:
    SOFA_CLASS(SOFA_TEMPLATE2(ReferenceFrictionContact,M1,M2),Base);
    ReferenceFrictionContact(M1*a,M2*b,sofa::core::collision::Intersection*i):Base(a,b,i){}
protected:
    void activateMappers()override{
        if(!this->m_constraint){
            auto* a=this->mapper1.createMapping(this->getName().c_str());
            auto* b=this->selfCollision?a:this->mapper2.createMapping(this->getName().c_str());
            this->m_constraint=sofa::core::objectmodel::New<ReferenceUnilateralContact>(a,b);
            this->m_constraint->setName(this->getName());
            this->setInteractionTags(a,b);this->m_constraint->setCustomTolerance(this->tol.getValue());
        }
        Base::activateMappers();
    }
};
using V=sofa::defaulttype::Vec3Types;
using Triangle=sofa::component::collision::TriangleCollisionModel<V>;
using Point=sofa::component::collision::PointCollisionModel<V>;
using Line=sofa::component::collision::LineCollisionModel<V>;
inline sofa::helper::Creator<sofa::core::collision::Contact::Factory,ReferenceFrictionContact<Triangle,Point>> pointContact("ReferenceFrictionContactConstraint",true);
inline sofa::helper::Creator<sofa::core::collision::Contact::Factory,ReferenceFrictionContact<Triangle,Line>> lineContact("ReferenceFrictionContactConstraint",true);
}
extern "C" int referenceContactUnitTest(double*out){
    using namespace mcr::contact_reference_test;
    using MO=sofa::component::container::MechanicalObject<V>;
    auto q=sofa::core::objectmodel::New<MO>();auto p=sofa::core::objectmodel::New<MO>();q->resize(1);p->resize(1);
    // Synthetic initial data only. This function is never called on a real replay state.
    for(auto id:{sofa::core::VecCoordId::position(),sofa::core::VecCoordId::freePosition()}){
        auto*d=q->write(id);auto*w=d->beginEdit();w->resize(1);(*w)[0]=V::Coord(0,0,0);d->endEdit();
        d=p->write(id);w=d->beginEdit();w->resize(1);(*w)[0]=V::Coord(id==sofa::core::VecCoordId::position()?0.00025:0.0001,0,0);d->endEdit();
    }
    auto c=sofa::core::objectmodel::New<ReferenceUnilateralContact>(q.get(),p.get());
    c->addContact(0.01,V::Deriv(1,0,0),V::Coord(0.001,0,0),V::Coord(0,0,0),0.0002,0,0,0,0);c->setUnitRow();
    sofa::linearalgebra::FullVector<double> v(3);
    sofa::core::ConstraintParams params;
    auto run=[&](bool enabled,bool current,int k){consistentReference=enabled;params.setX(sofa::core::ConstMultiVecCoordId(current?sofa::core::ConstVecCoordId::position():sofa::core::ConstVecCoordId::freePosition()));
        c->getConstraintViolation(&params,&v,*params.readX(q.get()),*params.readX(p.get()),*params.readV(q.get()),*params.readV(p.get()));
        for(int j=0;j<3;++j)out[k+j]=v.element(j);};
    run(false,false,0);run(true,false,3);run(true,true,6);consistentReference=false;
    return 0;
}
'''
(HERE/'native/ReferenceContact.h').write_text(header)
solver=(OLD/'native/TestOnlyRelinearizingGenericConstraintSolver.cpp').read_text()
solver='#include <iomanip>\n#include "ReferenceContact.h"\n'+solver
solver=solver.replace(': d_auditPath(',': d_consistent(initData(&d_consistent, false, "consistentContactReference", "Use native cParams reference coordinates in contact violation"))\n        , d_auditPath(',1)
solver=solver.replace('bool r=GenericSolver::buildSystem(p,x,v);audit("built",p);return r;','mcr::contact_reference_test::consistentReference=d_consistent.getValue();\n        bool r=GenericSolver::buildSystem(p,x,v);\n        mcr::contact_reference_test::consistentReference=false;\n        audit("built",p);return r;')
solver=solver.replace('    sofa::core::objectmodel::Data<std::string> d_auditPath;','    sofa::core::objectmodel::Data<bool> d_consistent;\n    sofa::core::objectmodel::Data<std::string> d_auditPath;')
solver=solver.replace('if(cs->getName()=="CollisionDOFs")','if(cs->getSize()<500)')
solver=solver.replace('vec(o,p->readX(cs)->getValue(),3);o<<"}";', '''vec(o,p->readX(cs)->getValue(),3);
            o<<",\\"applied_dx\\":";vec(o,cs->read(p->dx().getId(cs))->getValue(),3);
            o<<",\\"jacobian\\":[";bool firstJ=true;const auto& CJ=p->readJ(cs)->getValue();
            for(auto ri=CJ.begin();ri!=CJ.end();++ri)for(auto ci=ri.begin();ci!=ri.end();++ci){
                if(!firstJ)o<<",";firstJ=false;o<<"["<<ri.index()<<","<<ci.index();
                for(int k=0;k<3;++k)o<<","<<ci.val()[k];o<<"]";
            }o<<"]}";''')
solver=solver.replace('        if(current_cp){int n=', '''        std::vector<mcr::contact_reference_test::ReferenceUnilateralContact*> contacts;
        getContext()->getObjects(&contacts,sofa::core::objectmodel::BaseContext::SearchDown);
        o<<",\\"contact_geometry\\":[";bool firstContact=true;
        for(auto*c:contacts){if(!firstContact)o<<",";firstContact=false;o<<c->lastGeometry;}o<<"]";
        if(current_cp){int n=''')
solver=solver.replace('MCRRepeatSolveConsistency','MCRContactReferenceConsistency')
(HERE/'native/TestOnlyRelinearizingGenericConstraintSolver.cpp').write_text(solver)
cmake=(OLD/'native/CMakeLists.txt').read_text().replace('MCRRepeatSolveConsistency','MCRContactReferenceConsistency')
cmake+='\ntarget_link_libraries(MCRContactReferenceConsistency PRIVATE "${SOFA_PREFIX}/lib/libSofaMeshCollision.so" "${SOFA_PREFIX}/lib/libSofaBaseMechanics.so")\n'
cmake+='target_include_directories(MCRContactReferenceConsistency PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/../results/build_deps/eigen-git-mirror-3.3.7")\n'
(HERE/'native/CMakeLists.txt').write_text(cmake)
run=(OLD/'run_case.py').read_text()
run=run.replace("[('P1',1),('P2',2)] if CASE['worker']==9 else [('P1',1)]","[('A',1),('B',2)] if CASE['worker']==9 else [('A',1)]")
run=run.replace("self.branch=('P3' if CASE['worker']==9 else 'P2'); self.passes=(3 if CASE['worker']==9 else 2)","self.branch='C'; self.passes=2")
run=run.replace('self.native.relinearizationEnabled.value=True','self.native.relinearizationEnabled.value=True\n            self.native.consistentContactReference.value=self.branch==\'C\'')
run=run.replace("'expected_animate_calls':3-FIRST","'expected_animate_calls':1").replace('self.target_animation_calls-(3-FIRST)','self.target_animation_calls-1')
run=run.replace("if not summary['nonpenetration_pass'] or summary['error'] or upcoming==2: raise StopTarget()","raise StopTarget()")
run=run.replace('libMCRRepeatSolveConsistency.so','libMCRContactReferenceConsistency.so')
run=run.replace('Sofa.Simulation.init=base._patched_sim_init','''def consistent_scene_init(root,*a,**kw):
        response=root.getObject('Response')
        if response is None: raise RuntimeError('CONTACT_MANAGER_NOT_FOUND')
        response.response.value='ReferenceFrictionContactConstraint'
        return base._patched_sim_init(root,*a,**kw)
    Sofa.Simulation.init=consistent_scene_init''')
run=run.replace("        self.substeps.append(summary); self.last_committed=q.copy()",'''        mapped=[]
        def collect(node):
            for obj in node.objects:
                if obj.getClassName()!='MechanicalObject': continue
                values=np.asarray(obj.position.value)
                if values.ndim==2 and values.shape[1]==3 and len(values)<500:
                    mapped.append({'path':str(obj.getLinkPath()).lstrip('@'),'position':values.tolist()})
            for child in node.children: collect(child)
        collect(self._sofa_root_node)
        summary['final_mapped']=mapped
        self.substeps.append(summary); self.last_committed=q.copy()''')
(HERE/'run_case.py').write_text(run)
print('ISOLATED_SOURCES_CREATED',flush=True)
