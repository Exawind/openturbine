#pragma once

#include "src/gebt_poc/types.hpp"
#include "src/gen_alpha_poc/time_stepper.h"
#include "src/gen_alpha_poc/utilities.h"

namespace openturbine::gebt_poc {

/// Abstract base class to provide problem-specific residual vector and iteration matrix
/// for the generalized-alpha solver
class LinearizationParameters {
public:
    virtual ~LinearizationParameters() = default;

    /// Interface for calculating the residual vector for the problem
    virtual void ResidualVector(
        LieGroupFieldView::const_type /* gen_coords */,
        LieAlgebraFieldView::const_type /* velocity */,
        LieAlgebraFieldView::const_type /* acceleration */,
        View1D::const_type /* lagrange_multipliers */,
        const gen_alpha_solver::TimeStepper& /* time_stepper */, View1D /* residual_vector */
    ) = 0;

    /// Interface for calculating the iteration matrix for the problem
    virtual void IterationMatrix(
        double /* h */, double /* BetaPrime */, double /* GammaPrime */,
        LieGroupFieldView::const_type /* gen_coords */,
        LieAlgebraFieldView::const_type /* delta_gen_coords */,
        LieAlgebraFieldView::const_type /* velocity */,
        LieAlgebraFieldView::const_type /* acceleration */, View1D::const_type /* lagrange_mults */,
        View2D /* iteration_matrix */
    ) = 0;
};

/// Defines a unity residual vector and identity iteration matrix
class UnityLinearizationParameters : public LinearizationParameters {
public:
    UnityLinearizationParameters(){};

    /// Returns a unity residual vector
    virtual void ResidualVector(
        LieGroupFieldView::const_type gen_coords, LieAlgebraFieldView::const_type velocity,
        LieAlgebraFieldView::const_type acceleration, View1D::const_type lagrange_mults,
        const gen_alpha_solver::TimeStepper& time_stepper, View1D residual_vector
    ) override;

    /// Returns an identity iteration matrix
    virtual void IterationMatrix(
        double h, double BetaPrime, double GammaPrime, LieGroupFieldView::const_type gen_coords,
        LieAlgebraFieldView::const_type delta_gen_coords, LieAlgebraFieldView::const_type velocity,
        LieAlgebraFieldView::const_type acceleration, View1D::const_type lagrange_mults,
        View2D iteration_matrix
    ) override;
};

}  // namespace openturbine::gebt_poc
