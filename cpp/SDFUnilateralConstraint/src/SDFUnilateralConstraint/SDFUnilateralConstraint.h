#pragma once

#include <SDFUnilateralConstraint/config.h>

#include <sofa/core/behavior/Constraint.h>
#include <sofa/core/behavior/ConstraintResolution.h>
#include <sofa/core/behavior/MechanicalState.h>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/type/Vec.h>
#include <sofa/type/vector.h>

#include <vector>

namespace mcr::constraint
{

class SDFUnilateralConstraintResolution final
    : public sofa::core::behavior::ConstraintResolution
{
public:
    SDFUnilateralConstraintResolution()
        : sofa::core::behavior::ConstraintResolution(1)
    {
    }

    void resolution(
        int line,
        double** w,
        double* d,
        double* force,
        double* dfree) override;
};

class SDFUNILATERALCONSTRAINT_API SDFUnilateralConstraint final
    : public sofa::core::behavior::Constraint<sofa::defaulttype::Vec3Types>
{
public:
    using DataTypes = sofa::defaulttype::Vec3Types;
    using Inherit = sofa::core::behavior::Constraint<DataTypes>;
    using MechanicalState = sofa::core::behavior::MechanicalState<DataTypes>;
    using VecCoord = DataTypes::VecCoord;
    using VecDeriv = DataTypes::VecDeriv;
    using Coord = DataTypes::Coord;
    using Deriv = DataTypes::Deriv;
    using MatrixDeriv = DataTypes::MatrixDeriv;
    using MatrixDerivRowIterator = MatrixDeriv::RowIterator;
    using DataVecCoord = sofa::core::objectmodel::Data<VecCoord>;
    using DataVecDeriv = sofa::core::objectmodel::Data<VecDeriv>;
    using DataMatrixDeriv = sofa::core::objectmodel::Data<MatrixDeriv>;
    using Real = DataTypes::Real;\n    using Vec3 = sofa::type::Vec<3, Real>;

    SOFA_CLASS(
        SDFUnilateralConstraint,
        SOFA_TEMPLATE(sofa::core::behavior::Constraint, sofa::defaulttype::Vec3Types));

    explicit SDFUnilateralConstraint(MechanicalState* object = nullptr);
    ~SDFUnilateralConstraint() override = default;

    void init() override;

    void buildConstraintMatrix(
        const sofa::core::ConstraintParams* cParams,
        DataMatrixDeriv& c_d,
        unsigned int& cIndex,
        const DataVecCoord& x) override;

    void getConstraintViolation(
        const sofa::core::ConstraintParams* cParams,
        sofa::linearalgebra::BaseVector* resV,
        const DataVecCoord& x,
        const DataVecDeriv& v) override;

    void getConstraintResolution(
        const sofa::core::ConstraintParams* cParams,
        std::vector<sofa::core::behavior::ConstraintResolution*>& resTab,
        unsigned int& offset) override;

private:
    sofa::core::objectmodel::Data<bool> d_enabled;
    sofa::core::objectmodel::Data<sofa::type::vector<unsigned int>> d_indices0;
    sofa::core::objectmodel::Data<sofa::type::vector<unsigned int>> d_indices1;
    sofa::core::objectmodel::Data<sofa::type::vector<double>> d_weights0;
    sofa::core::objectmodel::Data<sofa::type::vector<double>> d_weights1;
    sofa::core::objectmodel::Data<sofa::type::vector<Vec3>> d_normals;
    sofa::core::objectmodel::Data<sofa::type::vector<Vec3>> d_anchors;
    sofa::core::objectmodel::Data<sofa::type::vector<double>> d_sourceClearances;
    sofa::core::objectmodel::Data<unsigned int> d_activeCount;

    sofa::type::vector<unsigned int> m_activeSlots;
    sofa::type::vector<unsigned int> m_constraintIds;
};

} // namespace mcr::constraint
