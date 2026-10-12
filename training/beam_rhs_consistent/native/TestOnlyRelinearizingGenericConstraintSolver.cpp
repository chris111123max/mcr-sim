#include <iomanip>
#include "ReferenceContact.h"
#include <SofaConstraint/GenericConstraintSolver.h>
#include <sofa/core/ConstraintParams.h>
#include <sofa/core/ExecParams.h>
#include <sofa/core/MechanicalParams.h>
#include <sofa/core/MultiVecId.h>
#include <sofa/core/ObjectFactory.h>
#include <sofa/core/objectmodel/ScriptEvent.h>
#include <sofa/simulation/mechanicalvisitor/MechanicalPropagateOnlyPositionAndVelocityVisitor.h>

#include <fstream>
#include <iomanip>
#include <sofa/core/behavior/MechanicalState.h>
#include <sofa/defaulttype/RigidTypes.h>
#include <sofa/defaulttype/VecTypes.h>
#include <algorithm>
#include <chrono>
#include <string>

namespace mcr::nonlinear_test
{

using GenericSolver = sofa::component::constraintset::GenericConstraintSolver;
using ConstraintParams = sofa::core::ConstraintParams;

class TestOnlyRelinearizingGenericConstraintSolver final : public GenericSolver
{
public:
    SOFA_CLASS(TestOnlyRelinearizingGenericConstraintSolver, GenericSolver);

    TestOnlyRelinearizingGenericConstraintSolver()
        : d_consistent(initData(&d_consistent, false, "consistentContactReference", "Use native cParams reference coordinates in contact violation"))
        , d_auditDetail(initData(&d_auditDetail, true, "auditDetail", "Full diagnostic arrays; false retains built/solved RHS and reference audit"))
        , d_auditPath(initData(&d_auditPath, std::string(""), "auditPath", "Read-only native audit JSONL path"))
        , d_enabled(initData(&d_enabled, false, "relinearizationEnabled",
                             "Run native same-substep relinearization when enabled."))
        , d_requestedPasses(initData(&d_requestedPasses, 1u, "requestedPasses",
                                     "Number of genuine native Generic solver passes (1..3)."))
        , d_nativeSolveCalls(initData(&d_nativeSolveCalls, 0u, "nativeSolveCalls",
                                      "Native GenericConstraintSolver invocations in the armed substep."))
        , d_completedStages(initData(&d_completedStages, 0u, "completedStages",
                                     "Number of native solver stages completed in the armed substep."))
        , d_mappingRefreshCount(initData(&d_mappingRefreshCount, 0u, "mappingRefreshCount",
                                         "Native mechanical mapping refreshes between same-step solver stages."))
        , d_solveMs(initData(&d_solveMs, 0.0, "lastNativeSolveWallMs", "steady_clock time around native Generic solveConstraint only."))
        , d_mapMs(initData(&d_mapMs, 0.0, "lastMappingRefreshWallMs", "steady_clock time around native mapping visitor only."))
        , d_solveTotalMs(initData(&d_solveTotalMs, 0.0, "nativeSolveTotalWallMs", "Total native solveConstraint wall time in current substep."))
        , d_mapTotalMs(initData(&d_mapTotalMs, 0.0, "mappingTotalWallMs", "Total native mapping wall time in current substep."))
        , d_eventStatus(initData(&d_eventStatus, std::string("IDLE"), "stageEventStatus",
                                 "Python row-rebuild acknowledgement for the current stage."))
        , d_status(initData(&d_status, std::string("IDLE"), "relinearizationStatus",
                            "Test-only same-substep solver orchestration status."))
    {
    }

    void solveConstraint(const ConstraintParams* cParams,
                         sofa::core::MultiVecId res1,
                         sofa::core::MultiVecId res2 =
                             sofa::core::MultiVecId::null()) override
    {
        if (!d_enabled.getValue())
        {
            GenericSolver::solveConstraint(cParams, res1, res2);
            return;
        }

        unsigned int passes =
            std::clamp(d_requestedPasses.getValue(), 1u, 3u);
        d_nativeSolveCalls.setValue(0u);
        d_completedStages.setValue(0u);
        d_mappingRefreshCount.setValue(0u);
        d_status.setValue("RUNNING");
        d_solveTotalMs.setValue(0.0);
        d_mapTotalMs.setValue(0.0);

        for (unsigned int stage = 1; stage <= passes; ++stage)
        {
            ConstraintParams stageParams(*cParams);
            if (stage > 1)
            {
                stageParams.setX(sofa::core::ConstMultiVecCoordId(res1));
                if (!res2.isNull())
                    stageParams.setV(sofa::core::ConstMultiVecDerivId(res2));
            }

            const auto solveStart = std::chrono::steady_clock::now();
            GenericSolver::solveConstraint(&stageParams, res1, res2);
            const double solveMs = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - solveStart).count();
            d_solveMs.setValue(solveMs);
            d_solveTotalMs.setValue(d_solveTotalMs.getValue() + solveMs);
            d_mapMs.setValue(0.0);
            d_nativeSolveCalls.setValue(d_nativeSolveCalls.getValue() + 1u);

            d_completedStages.setValue(stage);
            d_eventStatus.setValue("WAITING");
            const std::string eventName =
                "MCR_NONLINEAR_RELINEARIZATION_STAGE_" + std::to_string(stage);
            sofa::core::objectmodel::ScriptEvent event(nullptr, eventName.c_str());
            getContext()->propagateEvent(
                sofa::core::execparams::defaultInstance(), &event);

            passes = std::clamp(d_requestedPasses.getValue(), 1u, 2u);
            if (stage < passes)
            {
                // Refresh mapped position/velocity vectors in the same SOFA step.
                // This is native mapping propagation only; it does not integrate,
                // project, or overwrite an independent mechanical state.
                sofa::core::MultiVecCoordId xId(res1);
                sofa::core::MultiVecDerivId vId(res2);
                sofa::simulation::mechanicalvisitor::
                    MechanicalPropagateOnlyPositionAndVelocityVisitor propagate(
                        sofa::core::mechanicalparams::defaultInstance(),
                        getContext()->getTime(), xId, vId);
                const auto mapStart = std::chrono::steady_clock::now();
                audit("pre_map", &stageParams);
                propagate.execute(getContext());
                audit("post_map", &stageParams);
                const double mapMs = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - mapStart).count();
                d_mapMs.setValue(mapMs);
                d_mapTotalMs.setValue(d_mapTotalMs.getValue() + mapMs);

                d_mappingRefreshCount.setValue(d_mappingRefreshCount.getValue() + 1u);
            }

            if (stage < passes)
            {
                const std::string expected =
                    "ROWS_REBUILT_FOR_STAGE_" + std::to_string(stage + 1);
                if (d_eventStatus.getValue() != expected)
                {
                    d_status.setValue("RELINEARIZATION_CALLBACK_FAILED:" +
                                      d_eventStatus.getValue());
                    return;
                }
            }
            else
            {
                const std::string expected =
                    "CAPTURED_STAGE_" + std::to_string(stage);
                if (d_eventStatus.getValue() != expected)
                {
                    d_status.setValue("FINAL_CAPTURE_CALLBACK_FAILED:" +
                                      d_eventStatus.getValue());
                    return;
                }
            }
        }

        d_status.setValue("PASS_NATIVE_SAME_SUBSTEP_RELINEARIZATION");
    }


    using State=sofa::core::behavior::MechanicalState<sofa::defaulttype::Rigid3Types>;
    template<class V> void vec(std::ostream& o,const V& v,int n) {
        o << "["; for(size_t i=0;i<v.size();++i) {if(i)o<<",";o<<"[";
        for(int j=0;j<n;++j){if(j)o<<",";o<<v[i][j];}o<<"]";}o<<"]";
    }
    unsigned int auditMissing = 0;
    template<class D> void auditVec(std::ostream& o,const D* data,int n) {
        if (!data) { o << "null"; ++auditMissing; return; }
        vec(o,data->getValue(),n);
    }
    void audit(const char* phase,const ConstraintParams* p) {
        if(!d_enabled.getValue()||d_auditPath.getValue().empty())return;
        const bool detail=d_auditDetail.getValue();
        const bool built=std::string(phase)=="built";
        if(!detail && !built && std::string(phase)!="solved")return;
        auditMissing=0;
        std::vector<State*> rigidStates;getContext()->getObjects(&rigidStates,sofa::core::objectmodel::BaseContext::SearchDown);
        State* s=nullptr;for(auto* candidate:rigidStates)if(candidate->getName()=="DOFs")s=candidate;
        std::ofstream o(d_auditPath.getValue(),std::ios::app);o<<std::setprecision(17);
        o<<"{\"stage\":"<<(d_nativeSolveCalls.getValue()+(std::string(phase).find("map")!=std::string::npos?0:1))<<",\"phase\":\""<<phase<<"\"";
        if(s){
            o<<",\"beam_path\":\""<<s->getPathName()<<"\"";
            auto* xp=p->readX(s);auto* vp=p->readV(s);
            o<<",\"reference_aliases_position\":"<<(xp==s->read(sofa::core::ConstVecCoordId::position())?"true":"false");
            o<<",\"reference_aliases_free\":"<<(xp==s->read(sofa::core::ConstVecCoordId::freePosition())?"true":"false");
            if(detail){
            o<<",\"position\":";auditVec(o,s->read(sofa::core::ConstVecCoordId::position()),7);
            o<<",\"free_position\":";auditVec(o,s->read(sofa::core::ConstVecCoordId::freePosition()),7);
            o<<",\"reference_x\":";auditVec(o,xp,7);
            o<<",\"velocity\":";auditVec(o,s->read(sofa::core::ConstVecDerivId::velocity()),6);
            o<<",\"free_velocity\":";auditVec(o,s->read(sofa::core::ConstVecDerivId::freeVelocity()),6);
            o<<",\"reference_v\":";auditVec(o,vp,6);
            o<<",\"native_dx\":";auditVec(o,s->read(getDx().getId(s)),6);
            o<<",\"applied_dx\":";auditVec(o,s->read(p->dx().getId(s)),6);
            }
            o<<",\"jacobian\":[";bool first=true;auto* jd=p->readJ(s); if(jd){const auto& J=jd->getValue();
            for(auto ri=J.begin();ri!=J.end();++ri)for(auto ci=ri.begin();ci!=ri.end();++ci){
                if(!first)o<<",";first=false;o<<"["<<ri.index()<<","<<ci.index();
                for(int j=0;j<6;++j)o<<","<<ci.val()[j];o<<"]";
            }} else {++auditMissing;} o<<"]";
        }
        if(detail){
        using VS=sofa::core::behavior::MechanicalState<sofa::defaulttype::Vec3Types>;
        std::vector<VS*> states;getContext()->getObjects(&states,sofa::core::objectmodel::BaseContext::SearchDown);
        o<<",\"mapped\":[";bool firstState=true;
        for(auto* cs:states)if(cs->getSize()<500){
            if(!firstState)o<<",";firstState=false;o<<"{\"path\":\""<<cs->getPathName()<<"\",\"position\":";
            auditVec(o,cs->read(sofa::core::ConstVecCoordId::position()),3);
            o<<",\"free_position\":";auditVec(o,cs->read(sofa::core::ConstVecCoordId::freePosition()),3);
            o<<",\"reference_x\":";auditVec(o,p->readX(cs),3);
            o<<",\"applied_dx\":";auditVec(o,cs->read(p->dx().getId(cs)),3);
            o<<",\"jacobian\":[";bool firstJ=true;auto* cjd=p->readJ(cs); if(cjd){const auto& CJ=cjd->getValue();
            for(auto ri=CJ.begin();ri!=CJ.end();++ri)for(auto ci=ri.begin();ci!=ri.end();++ci){
                if(!firstJ)o<<",";firstJ=false;o<<"["<<ri.index()<<","<<ci.index();
                for(int k=0;k<3;++k)o<<","<<ci.val()[k];o<<"]";
            }} else {++auditMissing;} o<<"]}";
        }o<<"]";
        }
        std::vector<mcr::contact_reference_test::ReferenceUnilateralContact*> contacts;
        getContext()->getObjects(&contacts,sofa::core::objectmodel::BaseContext::SearchDown);
        o<<",\"contact_geometry\":[";bool firstContact=true;
        for(auto*c:contacts){if(!firstContact)o<<",";firstContact=false;o<<c->lastGeometry;}o<<"]";
        if(current_cp){int n=current_cp->getDimension();
            o<<",\"dimension\":"<<n<<",\"rhs\":[";
            for(int i=0;i<n;++i){if(i)o<<",";o<<current_cp->dFree[i];}o<<"],\"lambda\":[";
            for(int i=0;i<n;++i){if(i)o<<",";o<<current_cp->f[i];}o<<"]";
            if(detail){o<<",\"W\":[";
            for(int i=0;i<n;++i){if(i)o<<",";o<<"[";for(int j=0;j<n;++j){if(j)o<<",";o<<current_cp->getW()[i][j];}o<<"]";}o<<"]";}
            o<<",\"error\":"<<current_cp->currentError<<",\"iterations\":"<<current_cp->currentIterations;
        }o<<",\"unavailable_data_count\":"<<auditMissing<<"}\n";
    }
    bool prepareStates(const ConstraintParams*p,sofa::core::MultiVecId x,sofa::core::MultiVecId v) override {
        bool r=GenericSolver::prepareStates(p,x,v);audit("prepared",p);return r;
    }
    bool buildSystem(const ConstraintParams*p,sofa::core::MultiVecId x,sofa::core::MultiVecId v) override {
        mcr::contact_reference_test::consistentReference=d_consistent.getValue();
        mcr::contact_reference_test::captureContactGeometry=!d_auditPath.getValue().empty();
        bool r=GenericSolver::buildSystem(p,x,v);
        mcr::contact_reference_test::captureContactGeometry=false;
        mcr::contact_reference_test::consistentReference=false;
        audit("built",p);return r;
    }
    bool solveSystem(const ConstraintParams*p,sofa::core::MultiVecId x,sofa::core::MultiVecId v) override {
        bool r=GenericSolver::solveSystem(p,x,v);audit("solved",p);return r;
    }
    bool applyCorrection(const ConstraintParams*p,sofa::core::MultiVecId x,sofa::core::MultiVecId v) override {
        audit("pre_apply",p);bool r=GenericSolver::applyCorrection(p,x,v);audit("post_apply",p);return r;
    }

private:
    sofa::core::objectmodel::Data<bool> d_consistent;
    sofa::core::objectmodel::Data<bool> d_auditDetail;
    sofa::core::objectmodel::Data<std::string> d_auditPath;
    sofa::core::objectmodel::Data<double> d_solveMs;
    sofa::core::objectmodel::Data<double> d_mapMs;
    sofa::core::objectmodel::Data<double> d_solveTotalMs;
    sofa::core::objectmodel::Data<double> d_mapTotalMs;
    sofa::core::objectmodel::Data<bool> d_enabled;
    sofa::core::objectmodel::Data<unsigned int> d_requestedPasses;
    sofa::core::objectmodel::Data<unsigned int> d_nativeSolveCalls;
    sofa::core::objectmodel::Data<unsigned int> d_completedStages;
    sofa::core::objectmodel::Data<unsigned int> d_mappingRefreshCount;
    sofa::core::objectmodel::Data<std::string> d_eventStatus;
    sofa::core::objectmodel::Data<std::string> d_status;
};

int TestOnlyRelinearizingGenericConstraintSolverClass =
    sofa::core::RegisterObject(
        "Test-only Generic solver wrapper that rebuilds nonlinear rows and invokes "
        "the SOFA constraint solver repeatedly inside one solveConstraint call.")
        .add<TestOnlyRelinearizingGenericConstraintSolver>();

} // namespace mcr::nonlinear_test

extern "C"
{
void initExternalModule() {}
const char* getModuleName() { return "MCRContactReferenceConsistency"; }
const char* getModuleVersion() { return "0.1"; }
const char* getModuleLicense() { return "Test-only research code"; }
const char* getModuleDescription()
{
    return "Test-only native same-substep nonlinear constraint relinearization.";
}
}
