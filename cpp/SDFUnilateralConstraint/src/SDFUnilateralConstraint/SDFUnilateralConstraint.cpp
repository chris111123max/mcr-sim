#include <SDFUnilateralConstraint/SDFUnilateralConstraint.h>

#include <sofa/core/ObjectFactory.h>
#include <sofa/helper/logging/Messaging.h>

#include <algorithm>
#include <cmath>
#include <limits>

namespace mcr::constraint
{

void SDFUnilateralConstraintResolution::resolution(
    int line,
    double** w,
    double* d,
    double* force,
    double* /*dfree*/)
{
    const double diagonal = w[line][line];
    if (!std::isfinite(diagonal) || std::abs(diagonal) < 1e-18)
    {
        force[line] = 0.0;
        return;
    }

    // Same normal-contact complementarity update used by SOFA 21.12's
    // UnilateralConstraintResolution: lambda >= 0 and g >= 0.
    force[line] -= d[line] / diagonal;
    if (force[line] < 0.0 || !std::isfinite(force[line]))
        force[line] = 0.0;
}

SDFUnilateralConstraint::SDFUnilateralConstraint(MechanicalState* object)
    : Inherit(object)
    , d_enabled(initData(&d_enabled, false, "enabled",
        "Enable SDF unilateral inequality rows."))
    , d_indices0(initData(&d_indices0, "indices0",
        "First CollisionDOF index for each sampled point."))
    , d_indices1(initData(&d_indices1, "indices1",
        "Second CollisionDOF index for each sampled point."))
    , d_weights0(initData(&d_weights0, "weights0",
        "Barycentric weight of indices0."))
    , d_weights1(initData(&d_weights1, "weights1",
        "Barycentric weight of indices1."))
    , d_normals(initData(&d_normals, "normals",
        "Unit inward normals for g(x)>=0."))
    , d_anchors(initData(&d_anchors, "anchors",
        "Points on the catheter-center admissible boundary."))
    , d_sourceClearances(initData(&d_sourceClearances, "sourceClearances",
        "Diagnostic SDF clearances used to create the rows."))
    , d_activeCount(initData(&d_activeCount, 0u, "activeCount",
        "Number of unilateral rows built in the current solve."))
{
}

void SDFUnilateralConstraint::init()
{
    Inherit::init();
    this->mstate = dynamic_cast<MechanicalState*>(
        this->getContext()->getMechanicalState());

    if (!this->mstate)
        std::cerr
            << "[SDFUnilateralConstraint] Requires a Vec3 mechanical state "
            << "in the same node." << std::endl;
}

void SDFUnilateralConstraint::buildConstraintMatrix(
    const sofa::core::ConstraintParams* /*cParams*/,
    DataMatrixDeriv& c_d,
    unsigned int& cIndex,
    const DataVecCoord& x)
{
    m_activeSlots.clear();
    m_constraintIds.clear();
    d_activeCount.setValue(0u);

    if (!d_enabled.getValue() || !this->mstate)
        return;

    const auto& indices0 = d_indices0.getValue();
    const auto& indices1 = d_indices1.getValue();
    const auto& weights0 = d_weights0.getValue();
    const auto& weights1 = d_weights1.getValue();
    const auto& normals = d_normals.getValue();
    const auto& anchors = d_anchors.getValue();

    const std::size_t count = std::min(
        {indices0.size(), indices1.size(), weights0.size(), weights1.size(),
         normals.size(), anchors.size()});

    const auto& positions = x.getValue();
    MatrixDeriv& c = *c_d.beginEdit();

    for (std::size_t slot = 0; slot < count; ++slot)
    {
        const unsigned int i0 = indices0[slot];
        const unsigned int i1 = indices1[slot];
        if (i0 >= positions.size() || i1 >= positions.size())
            continue;

        Vec3 n = normals[slot];
        const Real norm = n.norm();
        if (!std::isfinite(static_cast<double>(norm)) || norm < 1e-12)
            continue;
        n /= norm;

        const double w0 = weights0[slot];
        const double w1 = weights1[slot];
        if (!std::isfinite(w0) || !std::isfinite(w1))
            continue;

        const unsigned int cid = cIndex++;
        MatrixDerivRowIterator row = c.writeLine(cid);

        const Deriv d0(
            static_cast<Real>(w0) * n[0],
            static_cast<Real>(w0) * n[1],
            static_cast<Real>(w0) * n[2]);
        row.addCol(i0, d0);

        if (i1 != i0 && std::abs(w1) > 1e-15)
        {
            const Deriv d1(
                static_cast<Real>(w1) * n[0],
                static_cast<Real>(w1) * n[1],
                static_cast<Real>(w1) * n[2]);
            row.addCol(i1, d1);
        }

        m_activeSlots.push_back(static_cast<unsigned int>(slot));
        m_constraintIds.push_back(cid);
    }

    c_d.endEdit();
    d_activeCount.setValue(static_cast<unsigned int>(m_activeSlots.size()));
}

void SDFUnilateralConstraint::getConstraintViolation(
    const sofa::core::ConstraintParams* /*cParams*/,
    sofa::linearalgebra::BaseVector* resV,
    const DataVecCoord& x,
    const DataVecDeriv& /*v*/)
{
    const auto& positions = x.getValue();
    const auto& indices0 = d_indices0.getValue();
    const auto& indices1 = d_indices1.getValue();
    const auto& weights0 = d_weights0.getValue();
    const auto& weights1 = d_weights1.getValue();
    const auto& normals = d_normals.getValue();
    const auto& anchors = d_anchors.getValue();

    const std::size_t rows = std::min(
        m_activeSlots.size(), m_constraintIds.size());

    for (std::size_t rowIndex = 0; rowIndex < rows; ++rowIndex)
    {
        const unsigned int slot = m_activeSlots[rowIndex];
        if (slot >= indices0.size() || slot >= indices1.size()
            || slot >= weights0.size() || slot >= weights1.size()
            || slot >= normals.size() || slot >= anchors.size())
            continue;

        const unsigned int i0 = indices0[slot];
        const unsigned int i1 = indices1[slot];
        if (i0 >= positions.size() || i1 >= positions.size())
            continue;

        Vec3 n = normals[slot];
        const Real norm = n.norm();
        if (norm < 1e-12)
            continue;
        n /= norm;

        const Coord sample =
            positions[i0] * static_cast<Real>(weights0[slot])
            + positions[i1] * static_cast<Real>(weights1[slot]);

        // g(x) = dot(sample - anchor, inward_normal).
        // Positive means inside the admissible lumen-center half-space.
        const double violation = static_cast<double>(
            (sample - anchors[slot]) * n);

        resV->set(m_constraintIds[rowIndex], violation);
    }
}

void SDFUnilateralConstraint::getConstraintResolution(
    const sofa::core::ConstraintParams* /*cParams*/,
    std::vector<sofa::core::behavior::ConstraintResolution*>& resTab,
    unsigned int& offset)
{
    for (std::size_t i = 0; i < m_activeSlots.size(); ++i)
        resTab[offset++] = new SDFUnilateralConstraintResolution();
}

int SDFUnilateralConstraintClass =
    sofa::core::RegisterObject(
        "SDF-driven unilateral g(x)>=0 constraint for mapped catheter samples.")
        .add<SDFUnilateralConstraint>();

} // namespace mcr::constraint
