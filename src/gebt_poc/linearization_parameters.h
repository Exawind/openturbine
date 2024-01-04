#pragma once

#include "src/gen_alpha_poc/time_stepper.h"
#include "src/gen_alpha_poc/utilities.h"

namespace openturbine::gebt_poc {

/// Abstract base class to provide problem-specific residual vector and iteration matrix
/// for the generalized-alpha solver
class LinearizationParameters {
public:
    static constexpr size_t kNumberOfLieGroupComponents = 7;
    static constexpr size_t kNumberOfLieAlgebraComponents = 6;

    virtual ~LinearizationParameters() = default;

    /// Interface for calculating the residual vector for the problem
    virtual void ResidualVector(
        Kokkos::View<double* [kNumberOfLieGroupComponents]> /* gen_coords */,
        Kokkos::View<double* [kNumberOfLieAlgebraComponents]> /* velocity */,
        Kokkos::View<double* [kNumberOfLieAlgebraComponents]> /* acceleration */,
        Kokkos::View<double*> /* lagrange_multipliers */,
        Kokkos::View<double*> /* residual_vector */,
        const gen_alpha_solver::TimeStepper& /* time_stepper */
    ) = 0;

    /// Interface for calculating the iteration matrix for the problem
    virtual void IterationMatrix(
        const double& /* h */, const double& /* BetaPrime */, const double& /* GammaPrime */,
        Kokkos::View<double* [kNumberOfLieGroupComponents]> /* gen_coords */,
        Kokkos::View<double* [kNumberOfLieAlgebraComponents]> /* delta_gen_coords */,
        Kokkos::View<double* [kNumberOfLieAlgebraComponents]> /* velocity */,
        Kokkos::View<double* [kNumberOfLieAlgebraComponents]> /* acceleration */,
        Kokkos::View<double*> /* lagrange_mults */, Kokkos::View<double**> /* iteration_matrix */
    ) = 0;
};

/// Defines a unity residual vector and identity iteration matrix
class UnityLinearizationParameters : public LinearizationParameters {
public:
    UnityLinearizationParameters(){};

    /// Returns a unity residual vector
    virtual void ResidualVector(
        Kokkos::View<double* [kNumberOfLieGroupComponents]> gen_coords,
        Kokkos::View<double* [kNumberOfLieAlgebraComponents]> velocity,
        Kokkos::View<double* [kNumberOfLieAlgebraComponents]> acceleration,
        Kokkos::View<double*> lagrange_mults, Kokkos::View<double*> residual_vector,
        const gen_alpha_solver::TimeStepper& time_stepper
    ) override;

    /// Returns an identity iteration matrix
    virtual void IterationMatrix(
        const double& h, const double& BetaPrime, const double& GammaPrime,
        Kokkos::View<double* [kNumberOfLieGroupComponents]> gen_coords,
        Kokkos::View<double* [kNumberOfLieAlgebraComponents]> delta_gen_coords,
        Kokkos::View<double* [kNumberOfLieAlgebraComponents]> velocity,
        Kokkos::View<double* [kNumberOfLieAlgebraComponents]> acceleration,
        Kokkos::View<double*> lagrange_mults, Kokkos::View<double**> iteration_matrix
    ) override;
};

}  // namespace openturbine::gebt_poc
