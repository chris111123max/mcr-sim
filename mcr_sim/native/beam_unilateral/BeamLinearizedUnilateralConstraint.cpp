#include <sofa/core/ObjectFactory.h>
#include <sofa/core/behavior/Constraint.h>
#include <sofa/core/behavior/ConstraintResolution.h>
#include <sofa/core/behavior/MechanicalState.h>
#include <sofa/defaulttype/RigidTypes.h>
#include <sofa/type/Vec.h>
#include <sofa/type/vector.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>

namespace mcr::beamunilateral
{

using RigidTypes = sofa::defaulttype::Rigid3Types;
using MechanicalState = sofa::core::behavior::MechanicalState<RigidTypes>;
using VecCoord = RigidTypes::VecCoord;
using VecDeriv = RigidTypes::VecDeriv;
using Deriv = RigidTypes::Deriv;
using MatrixDeriv = RigidTypes::MatrixDeriv;
using MatrixDerivRowIterator = MatrixDeriv::RowIterator;
using DataVecCoord = sofa::core::objectmodel::Data<VecCoord>;
using DataVecDeriv = sofa::core::objectmodel::Data<VecDeriv>;
using DataMatrixDeriv = sofa::core::objectmodel::Data<MatrixDeriv>;
using Vec3 = sofa::type::Vec<3, double>;

class BeamLinearizedUnilateralResolution final
    : public sofa::core::behavior::ConstraintResolution
{
public:
    BeamLinearizedUnilateralResolution()
        : sofa::core::behavior::ConstraintResolution(1)
    {
    }

    void resolution(
        int line,
        double** w,
        double* d,
        double* force,
        double* /*dfree*/) override
    {
        const double diagonal = w[line][line];
        if (!std::isfinite(diagonal) || std::abs(diagonal) < 1e-18)
        {
            force[line] = 0.0;
            return;
        }

        force[line] -= d[line] / diagonal;
        if (force[line] < 0.0 || !std::isfinite(force[line]))
            force[line] = 0.0;
    }
};

class BeamLinearizedUnilateralConstraint final
    : public sofa::core::behavior::Constraint<RigidTypes>
{
public:
    using Inherit = sofa::core::behavior::Constraint<RigidTypes>;

    SOFA_CLASS(BeamLinearizedUnilateralConstraint, Inherit);

    explicit BeamLinearizedUnilateralConstraint(MechanicalState* object = nullptr)
        : Inherit(object)
        , d_enabled(initData(
              &d_enabled, false, "enabled",
              "Enable linearized Beam/SDF unilateral rows."))
        , d_rowOffsets(initData(
              &d_rowOffsets, "rowOffsets",
              "CSR row offsets into dofIndices/Jacobian arrays; size rows+1."))
        , d_dofIndices(initData(
              &d_dofIndices, "dofIndices",
              "Rigid3 Beam DOF index for each nonzero Jacobian block."))
        , d_linearJacobian(initData(
              &d_linearJacobian, "linearJacobian",
              "d(clearance)/d(translation) Vec3 blocks for each nonzero."))
        , d_angularJacobian(initData(
              &d_angularJacobian, "angularJacobian",
              "d(clearance)/d(rotation-vector) Vec3 blocks for each nonzero."))
        , d_freeViolations(initData(
              &d_freeViolations, "freeViolations",
              "Linearized g_free = clearance - requested_margin for each row."))
        , d_sourceClearances(initData(
              &d_sourceClearances, "sourceClearances",
              "Dense Beam/SDF clearance used to create each row."))
        , d_activeCount(initData(
              &d_activeCount, 0u, "activeCount",
              "Number of unilateral rows built in the current constraint solve."))
    {
    }

    void init() override
    {
        Inherit::init();
        this->mstate = dynamic_cast<MechanicalState*>(
            this->getContext()->getMechanicalState());
        if (!this->mstate)
            std::cerr
                << "[BeamLinearizedUnilateralConstraint] Requires a Rigid3 "
                << "MechanicalState in the same node." << std::endl;
    }

    void buildConstraintMatrix(
        const sofa::core::ConstraintParams* /*cParams*/,
        DataMatrixDeriv& c_d,
        unsigned int& cIndex,
        const DataVecCoord& x) override
    {
        m_constraintIds.clear();
        m_activeRows.clear();
        d_activeCount.setValue(0u);

        if (!d_enabled.getValue() || !this->mstate)
            return;

        const auto& offsets = d_rowOffsets.getValue();
        const auto& indices = d_dofIndices.getValue();
        const auto& linear = d_linearJacobian.getValue();
        const auto& angular = d_angularJacobian.getValue();
        const auto& violations = d_freeViolations.getValue();
        const auto& positions = x.getValue();

        if (offsets.size() < 2)
            return;

        const std::size_t rowCount =
            std::min<std::size_t>(violations.size(), offsets.size() - 1);
        const std::size_t nnz =
            std::min({indices.size(), linear.size(), angular.size()});

        MatrixDeriv& c = *c_d.beginEdit();

        for (std::size_t r = 0; r < rowCount; ++r)
        {
            const std::size_t begin =
                std::min<std::size_t>(offsets[r], nnz);
            const std::size_t end =
                std::min<std::size_t>(offsets[r + 1], nnz);
            if (end <= begin)
                continue;

            const unsigned int cid = cIndex++;
            MatrixDerivRowIterator row = c.writeLine(cid);
            bool wrote = false;

            for (std::size_t k = begin; k < end; ++k)
            {
                const unsigned int dof = indices[k];
                if (dof >= positions.size())
                    continue;

                const Vec3& jl = linear[k];
                const Vec3& ja = angular[k];
                if (!(std::isfinite(jl[0]) && std::isfinite(jl[1])
                      && std::isfinite(jl[2]) && std::isfinite(ja[0])
                      && std::isfinite(ja[1]) && std::isfinite(ja[2])))
                    continue;

                Deriv deriv;
                deriv.getVCenter()[0] = jl[0];
                deriv.getVCenter()[1] = jl[1];
                deriv.getVCenter()[2] = jl[2];
                deriv.getVOrientation()[0] = ja[0];
                deriv.getVOrientation()[1] = ja[1];
                deriv.getVOrientation()[2] = ja[2];
                row.addCol(dof, deriv);
                wrote = true;
            }

            if (wrote)
            {
                m_constraintIds.push_back(cid);
                m_activeRows.push_back(static_cast<unsigned int>(r));
            }
        }

        c_d.endEdit();
        d_activeCount.setValue(
            static_cast<unsigned int>(m_activeRows.size()));
    }

    void getConstraintViolation(
        const sofa::core::ConstraintParams* /*cParams*/,
        sofa::linearalgebra::BaseVector* resV,
        const DataVecCoord& /*x*/,
        const DataVecDeriv& /*v*/) override
    {
        const auto& violations = d_freeViolations.getValue();
        const std::size_t rows =
            std::min(m_constraintIds.size(), m_activeRows.size());

        for (std::size_t i = 0; i < rows; ++i)
        {
            const unsigned int sourceRow = m_activeRows[i];
            if (sourceRow >= violations.size())
                continue;
            resV->set(m_constraintIds[i], violations[sourceRow]);
        }
    }

    void getConstraintResolution(
        const sofa::core::ConstraintParams* /*cParams*/,
        std::vector<sofa::core::behavior::ConstraintResolution*>& resTab,
        unsigned int& offset) override
    {
        for (std::size_t i = 0; i < m_activeRows.size(); ++i)
            resTab[offset++] = new BeamLinearizedUnilateralResolution();
    }

private:
    sofa::core::objectmodel::Data<bool> d_enabled;
    sofa::core::objectmodel::Data<sofa::type::vector<unsigned int>> d_rowOffsets;
    sofa::core::objectmodel::Data<sofa::type::vector<unsigned int>> d_dofIndices;
    sofa::core::objectmodel::Data<sofa::type::vector<Vec3>> d_linearJacobian;
    sofa::core::objectmodel::Data<sofa::type::vector<Vec3>> d_angularJacobian;
    sofa::core::objectmodel::Data<sofa::type::vector<double>> d_freeViolations;
    sofa::core::objectmodel::Data<sofa::type::vector<double>> d_sourceClearances;
    sofa::core::objectmodel::Data<unsigned int> d_activeCount;

    sofa::type::vector<unsigned int> m_constraintIds;
    sofa::type::vector<unsigned int> m_activeRows;
};

int BeamLinearizedUnilateralConstraintClass =
    sofa::core::RegisterObject(
        "Rigid3 Beam/SDF linearized unilateral g>=0 rows supplied by Python.")
        .add<BeamLinearizedUnilateralConstraint>();

} // namespace mcr::beamunilateral

extern "C"
{
void initExternalModule() {}
const char* getModuleName() { return "MCRBeamLinearizedUnilateral"; }
const char* getModuleVersion() { return "0.2"; }
const char* getModuleLicense() { return "Research code"; }
const char* getModuleDescription()
{
    return "Beam-level linearized SDF unilateral constraint.";
}
}
