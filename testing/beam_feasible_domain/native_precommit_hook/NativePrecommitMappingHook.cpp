#include <sofa/core/ObjectFactory.h>
#include <sofa/core/VecId.h>
#include <sofa/core/behavior/MechanicalState.h>
#include <sofa/core/objectmodel/BaseObject.h>
#include <sofa/core/objectmodel/Link.h>
#include <sofa/defaulttype/RigidTypes.h>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/helper/accessor.h>
#include <sofa/simulation/CollisionBeginEvent.h>
#include <sofa/simulation/mechanicalvisitor/MechanicalPropagateOnlyPositionAndVelocityVisitor.h>

#include <algorithm>
#include <cmath>
#include <string>

namespace mcr::beamfeasible
{

using RigidTypes = sofa::defaulttype::Rigid3Types;
using Vec3Types = sofa::defaulttype::Vec3Types;
using RigidState = sofa::core::behavior::MechanicalState<RigidTypes>;
using Vec3State = sofa::core::behavior::MechanicalState<Vec3Types>;

class BeamFeasibleNativePrecommitHook final : public sofa::core::objectmodel::BaseObject
{
public:
    SOFA_CLASS(BeamFeasibleNativePrecommitHook, sofa::core::objectmodel::BaseObject);

    using RigidVecCoord = RigidTypes::VecCoord;
    using RigidVecDeriv = RigidTypes::VecDeriv;
    using Vec3VecCoord = Vec3Types::VecCoord;

    using BeamLink = sofa::core::objectmodel::SingleLink<
        BeamFeasibleNativePrecommitHook,
        RigidState,
        sofa::core::objectmodel::BaseLink::FLAG_STOREPATH |
            sofa::core::objectmodel::BaseLink::FLAG_STRONGLINK>;

    using CollisionLink = sofa::core::objectmodel::SingleLink<
        BeamFeasibleNativePrecommitHook,
        Vec3State,
        sofa::core::objectmodel::BaseLink::FLAG_STOREPATH |
            sofa::core::objectmodel::BaseLink::FLAG_STRONGLINK>;

    BeamFeasibleNativePrecommitHook()
        : l_beamState(initLink("beamState", "Rigid3 Beam MechanicalState whose free state is replaced"))
        , l_collisionState(initLink("collisionState", "Mapped Vec3 collision state; diagnostic read only"))
        , d_candidateFreePosition(initData(
              &d_candidateFreePosition,
              "candidateFreePosition",
              "Accepted offline Rigid3 Beam free-position candidate"))
        , d_dt(initData(&d_dt, 0.005, "dt", "Physics substep dt"))
        , d_armed(initData(&d_armed, false, "armed", "Execute once on the next CollisionBeginEvent"))
        , d_fired(initData(&d_fired, false, "fired", "True after at least one armed event was handled"))
        , d_fireCount(initData(&d_fireCount, 0, "fireCount", "Number of armed CollisionBeginEvent injections completed"))
        , d_propagated(initData(&d_propagated, false, "propagated", "True after native mechanical mapping propagation"))
        , d_velocityCorrected(initData(&d_velocityCorrected, false, "velocityCorrected", "True after coherent freeVelocity correction"))
        , d_mappedChildMaxChangeMm(initData(
              &d_mappedChildMaxChangeMm, 0.0, "mappedChildMaxChangeMm",
              "Immediate mapped CollisionDOF free-position change, diagnostic only, mm"))
        , d_parentWriteMaxErrorMm(initData(
              &d_parentWriteMaxErrorMm, 0.0, "parentWriteMaxErrorMm",
              "Maximum parent Beam translation write error versus candidate, mm"))
        , d_capturedFreePosition(initData(
              &d_capturedFreePosition, "capturedFreePosition",
              "Live unsafe Rigid3 free_position captured before replacement"))
        , d_status(initData(&d_status, std::string("IDLE"), "status", "One-frame native-hook status"))
    {
        f_listening.setValue(true);
    }

    void init() override
    {
        Inherit1::init();
        if (!l_beamState.get())
        {
            d_status.setValue("INVALID_BEAM_STATE_LINK");
            msg_error() << "beamState link is not resolved";
            return;
        }
        if (!l_collisionState.get())
        {
            d_status.setValue("INVALID_COLLISION_STATE_LINK");
            msg_error() << "collisionState link is not resolved";
            return;
        }
        d_status.setValue("READY");
    }

    void handleEvent(sofa::core::objectmodel::Event* event) override
    {
        if (!sofa::simulation::CollisionBeginEvent::checkEventType(event))
            return;
        if (!d_armed.getValue())
            return;

        d_fired.setValue(true);
        d_fireCount.setValue(d_fireCount.getValue() + 1);
        d_armed.setValue(false);
        d_propagated.setValue(false);
        d_velocityCorrected.setValue(false);
        d_mappedChildMaxChangeMm.setValue(0.0);
        d_parentWriteMaxErrorMm.setValue(0.0);
        d_status.setValue("ARMED_EVENT_RUNNING");

        auto* beam = l_beamState.get();
        auto* collision = l_collisionState.get();
        if (!beam || !collision)
        {
            d_status.setValue("INVALID_STATE_LINK_AT_EVENT");
            return;
        }

        const auto candidate = d_candidateFreePosition.getValue();
        if (candidate.empty())
        {
            d_status.setValue("EMPTY_CANDIDATE");
            return;
        }

        RigidVecCoord freeBefore;
        RigidVecDeriv velocityBefore;
        Vec3VecCoord collisionBefore;

        {
            sofa::helper::ReadAccessor<sofa::core::objectmodel::Data<RigidVecCoord>> xfree =
                *beam->read(sofa::core::vec_id::read_access::freePosition);
            freeBefore.assign(xfree.begin(), xfree.end());
        }
        {
            sofa::helper::ReadAccessor<sofa::core::objectmodel::Data<RigidVecDeriv>> vfree =
                *beam->read(sofa::core::vec_id::read_access::freeVelocity);
            velocityBefore.assign(vfree.begin(), vfree.end());
        }
        {
            sofa::helper::ReadAccessor<sofa::core::objectmodel::Data<Vec3VecCoord>> child =
                *collision->read(sofa::core::vec_id::read_access::freePosition);
            collisionBefore.assign(child.begin(), child.end());
        }

        d_capturedFreePosition.setValue(freeBefore);

        if (candidate.size() != freeBefore.size() || velocityBefore.size() != freeBefore.size())
        {
            d_status.setValue("STATE_SIZE_MISMATCH");
            return;
        }

        const double dt = d_dt.getValue();
        if (!(std::isfinite(dt) && dt > 0.0))
        {
            d_status.setValue("INVALID_DT");
            return;
        }

        RigidVecDeriv correctedVelocity = velocityBefore;
        for (std::size_t i = 0; i < candidate.size(); ++i)
        {
            // Rigid3Types::coordDifference computes translation plus the SO(3)
            // quaternion difference candidate * live_free^{-1}. Quaternion
            // coefficients are never subtracted as Euclidean rotational DOFs.
            auto correction = RigidTypes::coordDifference(candidate[i], freeBefore[i]);
            correction *= (1.0 / dt);
            correctedVelocity[i] += correction;
        }

        {
            sofa::helper::WriteAccessor<sofa::core::objectmodel::Data<RigidVecCoord>> xfree =
                *beam->write(sofa::core::vec_id::write_access::freePosition);
            if (xfree.size() != candidate.size())
                xfree.resize(candidate.size());
            for (std::size_t i = 0; i < candidate.size(); ++i)
                xfree[i] = candidate[i];
        }

        {
            sofa::helper::WriteAccessor<sofa::core::objectmodel::Data<RigidVecDeriv>> vfree =
                *beam->write(sofa::core::vec_id::write_access::freeVelocity);
            if (vfree.size() != correctedVelocity.size())
                vfree.resize(correctedVelocity.size());
            for (std::size_t i = 0; i < correctedVelocity.size(); ++i)
                vfree[i] = correctedVelocity[i];
        }
        d_velocityCorrected.setValue(true);

        double maxWriteErrorM = 0.0;
        {
            sofa::helper::ReadAccessor<sofa::core::objectmodel::Data<RigidVecCoord>> written =
                *beam->read(sofa::core::vec_id::read_access::freePosition);
            if (written.size() != candidate.size())
            {
                d_status.setValue("PARENT_WRITE_SIZE_MISMATCH");
                return;
            }
            for (std::size_t i = 0; i < candidate.size(); ++i)
            {
                const auto delta = written[i].getCenter() - candidate[i].getCenter();
                maxWriteErrorM = std::max(maxWriteErrorM, static_cast<double>(delta.norm()));
            }
        }
        d_parentWriteMaxErrorMm.setValue(maxWriteErrorM * 1000.0);

        // Use SOFA's native mechanical top-down propagation, but explicitly on
        // freePosition/freeVelocity. Starting from InstrumentCombined keeps the
        // visitor in the catheter subtree and reaches MultiAdaptiveBeamMapping.
        sofa::simulation::mechanicalvisitor::MechanicalPropagateOnlyPositionAndVelocityVisitor visitor(
            sofa::core::mechanicalparams::defaultInstance(),
            this->getContext()->getTime(),
            sofa::core::vec_id::write_access::freePosition,
            sofa::core::vec_id::write_access::freeVelocity);
        visitor.execute(this->getContext());
        d_propagated.setValue(true);

        Vec3VecCoord collisionAfter;
        {
            sofa::helper::ReadAccessor<sofa::core::objectmodel::Data<Vec3VecCoord>> child =
                *collision->read(sofa::core::vec_id::read_access::freePosition);
            collisionAfter.assign(child.begin(), child.end());
        }

        if (collisionAfter.size() != collisionBefore.size())
        {
            d_status.setValue("COLLISION_STATE_SIZE_CHANGED");
            return;
        }

        double maxMappedChangeM = 0.0;
        for (std::size_t i = 0; i < collisionAfter.size(); ++i)
        {
            const auto delta = collisionAfter[i] - collisionBefore[i];
            maxMappedChangeM = std::max(maxMappedChangeM, static_cast<double>(delta.norm()));
        }
        d_mappedChildMaxChangeMm.setValue(maxMappedChangeM * 1000.0);

        if (!std::isfinite(maxMappedChangeM))
        {
            d_status.setValue("NON_FINITE_MAPPING_RESULT");
            return;
        }

        d_status.setValue("PASS_NATIVE_WRITE_AND_PROPAGATE");
    }

    BeamLink l_beamState;
    CollisionLink l_collisionState;

    sofa::core::objectmodel::Data<RigidVecCoord> d_candidateFreePosition;
    sofa::core::objectmodel::Data<double> d_dt;
    sofa::core::objectmodel::Data<bool> d_armed;
    sofa::core::objectmodel::Data<bool> d_fired;
    sofa::core::objectmodel::Data<int> d_fireCount;
    sofa::core::objectmodel::Data<bool> d_propagated;
    sofa::core::objectmodel::Data<bool> d_velocityCorrected;
    sofa::core::objectmodel::Data<double> d_mappedChildMaxChangeMm;
    sofa::core::objectmodel::Data<double> d_parentWriteMaxErrorMm;
    sofa::core::objectmodel::Data<RigidVecCoord> d_capturedFreePosition;
    sofa::core::objectmodel::Data<std::string> d_status;
};

int BeamFeasibleNativePrecommitHookClass =
    sofa::core::RegisterObject(
        "Test-only CollisionBeginEvent hook: replace a Rigid3 Beam free state "
        "with a supplied feasible candidate and re-propagate free vectors "
        "through native SOFA mechanical mappings before collision detection.")
        .add<BeamFeasibleNativePrecommitHook>();

} // namespace mcr::beamfeasible

extern "C"
{
void initExternalModule() {}
const char* getModuleName() { return "MCRBeamFeasibleHook"; }
const char* getModuleVersion() { return "0.1"; }
const char* getModuleLicense() { return "Test-only research code"; }
const char* getModuleDescription()
{
    return "Test-only native pre-commit mapping hook for mCR beam feasible-domain validation.";
}
}
