#pragma once

#include "src/rigid_pendulum_poc/utilities.h"

namespace openturbine::rigid_pendulum {

/// Abstract base class to provide problem-specific residual vector and iteration matrix
class LinearizationParameters {
public:
    virtual ~LinearizationParameters() = default;

    virtual Kokkos::View<double*> ResidualVector(
        const Kokkos::View<double*>, const Kokkos::View<double*>, const Kokkos::View<double*>, const Kokkos::View<double*>
    ) = 0;

    virtual Kokkos::View<double**> IterationMatrix(
        const double&, const double&, const double&, const Kokkos::View<double*>, const Kokkos::View<double*>,
        const Kokkos::View<double*>, const Kokkos::View<double*>, const Kokkos::View<double*>
    ) = 0;
};

/// Defines a unity residual vector and identity iteration matrix
class UnityLinearizationParameters : public LinearizationParameters {
public:
    UnityLinearizationParameters(){};

    virtual Kokkos::View<double*> ResidualVector(
        const Kokkos::View<double*>, const Kokkos::View<double*>, const Kokkos::View<double*>, const Kokkos::View<double*>
    ) override;

    virtual Kokkos::View<double**> IterationMatrix(
        const double&, const double&, const double&, const Kokkos::View<double*>, const Kokkos::View<double*>,
        const Kokkos::View<double*>, const Kokkos::View<double*>, const Kokkos::View<double*>
    ) override;
};

}  // namespace openturbine::rigid_pendulum
