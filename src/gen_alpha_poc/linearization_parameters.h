#pragma once

#include "src/gen_alpha_poc/utilities.h"

namespace openturbine::gen_alpha_solver {

/// Abstract base class to provide problem-specific residual vector and iteration matrix
/// for the generalized-alpha solver
class LinearizationParameters {
public:
    // Used to store a 1D Kokkos View of doubles on the host
    using HostView1D = Kokkos::View<double*, Kokkos::HostSpace>;

    // Used to store a 2D Kokkos View of doubles on the host
    using HostView2D = Kokkos::View<double**, Kokkos::HostSpace>;

    virtual ~LinearizationParameters() = default;

    /// Interface for calculating the residual vector for the problem
    virtual HostView1D ResidualVector(
        const HostView1D /* gen_coords */, const HostView1D /* velocity */,
        const HostView1D /* acceleration */, const HostView1D /* lagrange_multipliers */
    ) = 0;

    /// Interface for calculating the iteration matrix for the problem
    virtual HostView2D IterationMatrix(
        const double& /* h */, const double& /* BetaPrime */, const double& /* GammaPrime */,
        const HostView1D /* gen_coords */, const HostView1D /* delta_gen_coords */,
        const HostView1D /* velocity */, const HostView1D /* acceleration */,
        const HostView1D /* lagrange_mults */
    ) = 0;
};

/// Defines a unity residual vector and identity iteration matrix
class UnityLinearizationParameters : public LinearizationParameters {
public:
    UnityLinearizationParameters(){};

    /// Returns a unity residual vector
    virtual HostView1D ResidualVector(
        const HostView1D gen_coords, const HostView1D velocity, const HostView1D acceleration,
        const HostView1D lagrange_mults
    ) override;

    /// Returns an identity iteration matrix
    virtual HostView2D IterationMatrix(
        const double& h, const double& BetaPrime, const double& GammaPrime,
        const HostView1D gen_coords, const HostView1D delta_gen_coords, const HostView1D velocity,
        const HostView1D acceleration, const HostView1D lagrange_mults
    ) override;
};

}  // namespace openturbine::gen_alpha_solver
