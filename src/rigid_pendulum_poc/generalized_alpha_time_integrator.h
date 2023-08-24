#pragma once

#include "src/rigid_pendulum_poc/linearization_parameters.h"
#include "src/rigid_pendulum_poc/state.h"
#include "src/rigid_pendulum_poc/time_integrator.h"
#include "src/rigid_pendulum_poc/time_stepper.h"

namespace openturbine::rigid_pendulum {

/// @brief A time integrator class based on the generalized-alpha method
class GeneralizedAlphaTimeIntegrator : public TimeIntegrator {
public:
    static constexpr double kCONVERGENCETOLERANCE = 1e-12;

    GeneralizedAlphaTimeIntegrator(
        double alpha_f = 0.5, double alpha_m = 0.5, double beta = 0.25, double gamma = 0.5,
        TimeStepper time_stepper = TimeStepper(), bool precondition = false
    );

    /// Returns the type of the time integrator
    inline TimeIntegratorType GetType() const override {
        return TimeIntegratorType::kGENERALIZED_ALPHA;
    }

    /// Returns the alpha_f parameter
    inline double GetAlphaF() const { return kALPHA_F_; }

    /// Returns the alpha_m parameter
    inline double GetAlphaM() const { return kALPHA_M_; }

    /// Returns the beta parameter
    inline double GetBeta() const { return kBETA_; }

    /// Returns the gamma parameter
    inline double GetGamma() const { return kGAMMA_; }

    /// Returns a const reference to the time stepper
    inline const TimeStepper& GetTimeStepper() const { return time_stepper_; }

    /// Performs the time integration and returns a vector of States over the time steps
    virtual std::vector<State> Integrate(
        const State&, size_t, std::shared_ptr<LinearizationParameters> lin_params
    ) override;

    /*! Implements the solveTimeStep() algorithm of the Lie group based generalized-alpha
     *  method as described in Brüls, Cardona, and Arnold, "Lie group generalized-alpha time
     *  integration of constrained flexible multibody systems," 2012, Mechanism and
     *  Machine Theory, Vol 48, 121-137
     *  https://doi.org/10.1016/j.mechmachtheory.2011.07.017
     */
    std::tuple<State, Kokkos::View<double*>> AlphaStep(
        const State&, size_t, std::shared_ptr<LinearizationParameters> lin_params
    );

    /// Computes the updated generalized coordinates based on the non-linear update
    Kokkos::View<double*>
    UpdateGeneralizedCoordinates(const Kokkos::View<double*>, const Kokkos::View<double*>);

    /// Checks convergence of the non-linear solution based on the residuals
    bool CheckConvergence(const Kokkos::View<double*>);

    /// Returns the flag to indicate if the latest non-linear update has converged
    inline bool IsConverged() const { return is_converged_; }

private:
    const double kALPHA_F_;  //< Alpha_f coefficient of the generalized-alpha method
    const double kALPHA_M_;  //< Alpha_m coefficient of the generalized-alpha method
    const double kBETA_;     //< Beta coefficient of the generalized-alpha method
    const double kGAMMA_;    //< Gamma coefficient of the generalized-alpha method

    bool is_converged_;         //< Flag to indicate if the latest non-linear update has converged
    TimeStepper time_stepper_;  //< Time stepper object to perform the time integration
    bool precondition_;         //< Flag to indicate if the iteration matrix is preconditioned
};

}  // namespace openturbine::rigid_pendulum
