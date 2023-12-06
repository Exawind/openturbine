#pragma once

#include "src/gen_alpha_poc/utilities.h"

namespace openturbine::gen_alpha_solver {

/// Abstract base class to provide problem-specific residual vector and iteration matrix
/// for the generalized-alpha solver
class LinearizationParameters {
public:
    virtual ~LinearizationParameters() = default;

    /// Interface for calculating the residual vector for the problem
    virtual Kokkos::View<double*> ResidualVector(
        const Kokkos::View<double*> /* gen_coords */, const Kokkos::View<double*> /* velocity */,
        const Kokkos::View<double*> /* acceleration */,
        const Kokkos::View<double*> /* lagrange_multipliers */
    ) = 0;

    /// Interface for calculating the iteration matrix for the problem
    virtual Kokkos::View<double**> IterationMatrix(
        const double& /* h */, const double& /* BetaPrime */, const double& /* GammaPrime */,
        const Kokkos::View<double*> /* gen_coords */,
        const Kokkos::View<double*> /* delta_gen_coords */,
        const Kokkos::View<double*> /* velocity */, const Kokkos::View<double*> /* acceleration */,
        const Kokkos::View<double*> /* lagrange_mults */
    ) = 0;
};

/// Defines a unity residual vector and identity iteration matrix
class UnityLinearizationParameters : public LinearizationParameters {
public:
    UnityLinearizationParameters(){};

    /// Returns a unity residual vector
    virtual Kokkos::View<double*> ResidualVector(
        const Kokkos::View<double*> gen_coords, const Kokkos::View<double*> velocity,
        const Kokkos::View<double*> acceleration, const Kokkos::View<double*> lagrange_mults
    ) override;

    /// Returns an identity iteration matrix
    virtual Kokkos::View<double**> IterationMatrix(
        const double& h, const double& BetaPrime, const double& GammaPrime,
        const Kokkos::View<double*> gen_coords, const Kokkos::View<double*> delta_gen_coords,
        const Kokkos::View<double*> velocity, const Kokkos::View<double*> acceleration,
        const Kokkos::View<double*> lagrange_mults
    ) override;
};

}  // namespace openturbine::gen_alpha_solver
