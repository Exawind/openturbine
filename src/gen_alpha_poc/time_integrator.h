#pragma once

#include "src/gen_alpha_poc/linearization_parameters.h"
#include "src/gen_alpha_poc/state.h"

namespace openturbine::gen_alpha_solver {

// An enum class to indicate the type of time integrator
enum class TimeIntegratorType {
    kNewmarkBeta = 0,      //< Newmark-beta method
    kHHT = 1,              //< Hilber-Hughes-Taylor method
    kGeneralizedAlpha = 2  //< Generalized-alpha method
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

}  // namespace openturbine::gen_alpha_solver
