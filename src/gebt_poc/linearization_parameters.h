#pragma once

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
    virtual Kokkos::View<double*> ResidualVector(
        const Kokkos::View<double* [kNumberOfLieGroupComponents]> /* gen_coords */,
        const Kokkos::View<double* [kNumberOfLieAlgebraComponents]> /* velocity */,
        const Kokkos::View<double* [kNumberOfLieAlgebraComponents]> /* acceleration */,
        const Kokkos::View<double*> /* lagrange_multipliers */
    ) = 0;

    /// Interface for calculating the iteration matrix for the problem
    virtual Kokkos::View<double**> IterationMatrix(
        const double& /* h */, const double& /* BetaPrime */, const double& /* GammaPrime */,
        const Kokkos::View<double* [kNumberOfLieGroupComponents]> /* gen_coords */,
        const Kokkos::View<double* [kNumberOfLieAlgebraComponents]> /* delta_gen_coords */,
        const Kokkos::View<double* [kNumberOfLieAlgebraComponents]> /* velocity */,
        const Kokkos::View<double* [kNumberOfLieAlgebraComponents]> /* acceleration */,
        const Kokkos::View<double*> /* lagrange_mults */
    ) = 0;
};

/// Defines a unity residual vector and identity iteration matrix
class UnityLinearizationParameters : public LinearizationParameters {
public:
    UnityLinearizationParameters(){};

    /// Returns a unity residual vector
    virtual Kokkos::View<double*> ResidualVector(
        const Kokkos::View<double* [kNumberOfLieGroupComponents]> gen_coords,
        const Kokkos::View<double* [kNumberOfLieAlgebraComponents]> velocity,
        const Kokkos::View<double* [kNumberOfLieAlgebraComponents]> acceleration,
        const Kokkos::View<double*> lagrange_mults
    ) override;

    /// Returns an identity iteration matrix
    virtual Kokkos::View<double**> IterationMatrix(
        const double& h, const double& BetaPrime, const double& GammaPrime,
        const Kokkos::View<double* [kNumberOfLieGroupComponents]> gen_coords,
        const Kokkos::View<double* [kNumberOfLieAlgebraComponents]> delta_gen_coords,
        const Kokkos::View<double* [kNumberOfLieAlgebraComponents]> velocity,
        const Kokkos::View<double* [kNumberOfLieAlgebraComponents]> acceleration,
        const Kokkos::View<double*> lagrange_mults
    ) override;
};

}  // namespace openturbine::gebt_poc
