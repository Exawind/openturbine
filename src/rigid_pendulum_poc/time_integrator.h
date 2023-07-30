#pragma once

#include "src/rigid_pendulum_poc/state.h"
#include "src/rigid_pendulum_poc/utilities.h"

namespace openturbine::rigid_pendulum {

// An enum class to indicate the type of time integrator
enum class TimeIntegratorType {
    kNEWMARK_BETA = 0,   //< Newmark-beta method
    kHHT = 1,            //< Hilber-Hughes-Taylor method
    kGENERALIZED_ALPHA,  //< Generalized-alpha method
};

/// @brief An abstract class to provide a common interface for time integrators
class TimeIntegrator {
public:
    virtual ~TimeIntegrator() = default;

    /// Performs the time integration and returns a vector of States over the time steps
    virtual std::vector<State> Integrate(
        const State&, const MassMatrix&, const GeneralizedForces&, HostView1D,
        std::function<HostView2D(size_t)>, std::function<HostView1D(size_t)> vector
    ) = 0;

    /// Returns the type of the time integrator
    virtual TimeIntegratorType GetType() const = 0;
};

}  // namespace openturbine::rigid_pendulum
