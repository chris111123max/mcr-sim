#include <SofaConstraint/GenericConstraintSolver.h>
#include <sofa/core/ConstraintParams.h>
#include <sofa/core/ExecParams.h>
#include <sofa/core/MechanicalParams.h>
#include <sofa/core/MultiVecId.h>
#include <sofa/core/ObjectFactory.h>
#include <sofa/core/objectmodel/ScriptEvent.h>
#include <sofa/simulation/mechanicalvisitor/MechanicalPropagateOnlyPositionAndVelocityVisitor.h>

#include <algorithm>
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
        : d_enabled(initData(&d_enabled, false, "relinearizationEnabled",
                             "Run native same-substep relinearization when enabled."))
        , d_requestedPasses(initData(&d_requestedPasses, 1u, "requestedPasses",
                                     "Number of genuine native Generic solver passes (1..3)."))
        , d_nativeSolveCalls(initData(&d_nativeSolveCalls, 0u, "nativeSolveCalls",
                                      "Native GenericConstraintSolver invocations in the armed substep."))
        , d_completedStages(initData(&d_completedStages, 0u, "completedStages",
                                     "Number of native solver stages completed in the armed substep."))
        , d_mappingRefreshCount(initData(&d_mappingRefreshCount, 0u, "mappingRefreshCount",
                                         "Native mechanical mapping refreshes between same-step solver stages."))
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

        const unsigned int passes =
            std::clamp(d_requestedPasses.getValue(), 1u, 3u);
        d_nativeSolveCalls.setValue(0u);
        d_completedStages.setValue(0u);
        d_mappingRefreshCount.setValue(0u);
        d_status.setValue("RUNNING");

        for (unsigned int stage = 1; stage <= passes; ++stage)
        {
            ConstraintParams stageParams(*cParams);
            if (stage > 1)
            {
                stageParams.setX(sofa::core::ConstMultiVecCoordId(res1));
                if (!res2.isNull())
                    stageParams.setV(sofa::core::ConstMultiVecDerivId(res2));
            }

            GenericSolver::solveConstraint(&stageParams, res1, res2);
            d_nativeSolveCalls.setValue(d_nativeSolveCalls.getValue() + 1u);

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
                propagate.execute(getContext());

                d_mappingRefreshCount.setValue(d_mappingRefreshCount.getValue() + 1u);
            }

            d_completedStages.setValue(stage);
            d_eventStatus.setValue("WAITING");
            const std::string eventName =
                "MCR_NONLINEAR_RELINEARIZATION_STAGE_" + std::to_string(stage);
            sofa::core::objectmodel::ScriptEvent event(nullptr, eventName.c_str());
            getContext()->propagateEvent(
                sofa::core::execparams::defaultInstance(), &event);

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

private:
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
const char* getModuleName() { return "MCRNonlinearRelinearizationTest"; }
const char* getModuleVersion() { return "0.1"; }
const char* getModuleLicense() { return "Test-only research code"; }
const char* getModuleDescription()
{
    return "Test-only native same-substep nonlinear constraint relinearization.";
}
}
