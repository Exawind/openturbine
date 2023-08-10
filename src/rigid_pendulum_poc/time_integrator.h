#pragma once

#include "src/rigid_pendulum_poc/linearization_parameters.h"
#include "src/rigid_pendulum_poc/state.h"
#include "src/rigid_pendulum_poc/utilities.h"

namespace openturbine::rigid_pendulum {

using ResidualVector = std::function<
    HostView1D(const HostView1D, const HostView1D, const HostView1D, const HostView1D)>;

using IterationMatrix = std::function<HostView2D(
    const double&, const double&, const HostView1D, const HostView1D, const HostView1D,
    const double&, const HostView1D
)>;

// An enum class to indicate the type of time integrator
enum class TimeIntegratorType {
    kNewmarkBeta = 0,    //< Newmark-beta method
    kHHT = 1,            //< Hilber-Hughes-Taylor method
    kGeneralized_Alpha,  //< Generalized-alpha method
};

/// @brief An abstract class to provide a common interface for time integrators
class TimeIntegrator {
public:
    virtual ~TimeIntegrator() = default;

    /// Performs the time integration and returns a vector of States over the time steps
    virtual std::vector<State>
    Integrate(const State&, size_t, std::shared_ptr<LinearizationParameters>) = 0;

    /// Returns the type of the time integrator
    virtual TimeIntegratorType GetType() const = 0;
};

}  // namespace openturbine::rigid_pendulum
