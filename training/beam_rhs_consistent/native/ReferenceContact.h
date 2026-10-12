// Adapted small method from official SOFA v21.12.00, commit d300cc9af550afbd5c2167470676a5a7d47381a1.
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
inline thread_local bool captureContactGeometry = false;
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

    Real dfree = (Real)0.0;
    Real dfree_t = (Real)0.0;
    Real dfree_s = (Real)0.0;

    const unsigned int cSize = contacts.size();

    for (unsigned int i = 0; i < cSize; i++)
    {
        const Contact& c = contacts[i];

        // Compute dfree, dfree_t and d_free_s

        const Coord &Pfree =  PfreeVec[c.m2];
        const Coord &Qfree =  QfreeVec[c.m1];

        const Coord PPfree = Pfree - c.P;
        const Coord QQfree = Qfree - c.Q;

        const Real ref_dist = PPfree.norm() + QQfree.norm();

        dfree = dot(Pfree - Qfree, c.norm) - c.contactDistance;
        const Real delta = dot(c.P - c.Q, c.norm) - c.contactDistance;

        if ((helper::rabs(delta) < 0.00001 * ref_dist) && (helper::rabs(dfree) < 0.00001 * ref_dist))
        {
            dfree_t = dot(PPfree, c.t) - dot(QQfree, c.t);
            dfree_s = dot(PPfree, c.s) - dot(QQfree, c.s);
        }
        else if (helper::rabs(delta - dfree) > 0.001 * delta)
        {
            const Real dt = delta / (delta - dfree);

            if (dt > 0.0 && dt < 1.0)
            {
                const Coord Pt		= c.P * (1 - dt) + Pfree * dt;
                const Coord Qt		= c.Q * (1 - dt) + Qfree * dt;
                const Coord PtPfree = Pfree - Pt;
                const Coord QtQfree = Qfree - Qt;

                dfree_t = dot(PtPfree, c.t) - dot(QtQfree, c.t);
                dfree_s = dot(PtPfree, c.s) - dot(QtQfree, c.s);
            }
            else if (dfree < 0.0)
            {
                dfree_t = dot(PPfree, c.t) - dot(QQfree, c.t);
                dfree_s = dot(PPfree, c.s) - dot(QQfree, c.s);
            }
            else
            {
                dfree_t = 0;
                dfree_s = 0;
            }
        }
        else
        {
            dfree_t = dot(PPfree, c.t) - dot(QQfree, c.t);
            dfree_s = dot(PPfree, c.s) - dot(QQfree, c.s);
        }

        // Sets dfree in global violation vector

        v->set(c.id, dfree);

        c.dfree = dfree; // PJ : For isActive() method. Don't know if it's still usefull.

        if (c.mu > 0.0)
        {
            v->set(c.id + 1, dfree_t);
            v->set(c.id + 2, dfree_s);
        }
    }
    }
    void getPositionViolation(sofa::linearalgebra::BaseVector*v) override {
        const bool selected=consistentReference && reference;
        const auto& Q=selected?reference->readX(getMState1())->getValue():getMState1()->read(sofa::core::ConstVecCoordId::freePosition())->getValue();
        const auto& P=selected?reference->readX(getMState2())->getValue():getMState2()->read(sofa::core::ConstVecCoordId::freePosition())->getValue();
        if(selected)evaluateSelected(v,Q,P);else NativeContact::getPositionViolation(v);
        if (!captureContactGeometry) { lastGeometry="[]"; return; }
        std::ostringstream o;o<<std::setprecision(17)<<"[";bool first=true;
        for(const auto&c:contacts){if(!first)o<<",";first=false;
            o<<"{\"row\":"<<c.id<<",\"mu\":"<<c.mu<<",\"distance\":"<<c.contactDistance
             <<",\"selected_reference\":"<<(selected?"true":"false")<<",\"normal_gap\":"<<dot(P[c.m2]-Q[c.m1],c.norm)-c.contactDistance
             <<",\"m1\":"<<c.m1<<",\"m2\":"<<c.m2<<",\"state1\":\""<<getMState1()->getPathName()<<"\",\"state2\":\""<<getMState2()->getPathName()<<"\"";
            auto write=[&](const char*name,const auto&a){o<<",\""<<name<<"\":["<<a[0]<<","<<a[1]<<","<<a[2]<<"]";};
            write("P",P[c.m2]);write("Q",Q[c.m1]);write("normal",c.norm);write("tangent",c.t);write("second_tangent",c.s);
            write("anchor_P",c.P);write("anchor_Q",c.Q);
            const auto& PX=reference->readX(getMState2())->getValue();const auto& QX=reference->readX(getMState1())->getValue();
            write("cparams_P",PX[c.m2]);write("cparams_Q",QX[c.m1]);
            o<<",\"rhs_normal\":"<<v->element(c.id)<<",\"rhs_tangent\":"<<(c.mu>0?v->element(c.id+1):0)
             <<",\"rhs_second_tangent\":"<<(c.mu>0?v->element(c.id+2):0)<<"}";
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
