#pragma once

#include <Kokkos_Core.hpp>

#include "src/rigid_pendulum_poc/state.h"

namespace openturbine::rigid_pendulum {

HostView2D create_identity_matrix(size_t size);
HostView1D create_identity_vector(size_t size);

// An enum class to indicate the type of time integrator
enum class TimeIntegratorType {
    NEWMARK_BETA = 0,   //< Newmark-beta method
    HHT = 1,            //< Hilber-Hughes-Taylor method
    GENERALIZED_ALPHA,  //< Generalized-alpha method
};

/// @brief An abstract class to provide a common interface for time integrators
class TimeIntegrator {
public:
    virtual ~TimeIntegrator() = default;

    /// Performs the time integration and returns a vector of States over the time steps
    virtual std::vector<State>
    Integrate(const State&, std::function<HostView2D(size_t)>, std::function<HostView1D(size_t)>) = 0;

    /// Returns the type of the time integrator
    virtual TimeIntegratorType GetType() const = 0;
};

}  // namespace openturbine::rigid_pendulum
