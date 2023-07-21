#pragma once

#include "src/rigid_pendulum_poc/state.h"
#include "src/rigid_pendulum_poc/time_integrator.h"
#include "src/rigid_pendulum_poc/time_stepper.h"

namespace openturbine::rigid_pendulum {

/// @brief A time integrator class based on the generalized-alpha method
class GeneralizedAlphaTimeIntegrator : public TimeIntegrator {
public:
    static constexpr double kTOLERANCE = 1e-6;

    GeneralizedAlphaTimeIntegrator(
        double alpha_f = 0.5, double alpha_m = 0.5, double beta = 0.25, double gamma = 0.5,
        TimeStepper time_stepper = TimeStepper()
    );

    /// Returns the type of the time integrator
    inline TimeIntegratorType GetType() const override {
        return TimeIntegratorType::GENERALIZED_ALPHA;
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
        const State&, std::function<HostView2D(size_t)> iteration_matrix = create_identity_matrix,
        std::function<HostView1D(size_t)> residual_vector = create_identity_vector
    ) override;

    /*! @brief  Perform the alpha step of the generalized-alpha method as described
     *          in the paper by Arnold and Brüls (2007)
     *          https://link.springer.com/article/10.1007/s11044-007-9084-0
     * @param   state Current state of the system at the beginning of the time step
     * @return  Updated state of the system at the end of the time step
     */
    std::tuple<State, HostView1D>
    AlphaStep(const State&, std::function<HostView2D(size_t)>, std::function<HostView1D(size_t)>);

    /// Performs the linear update of the generalized-alpha method
    State UpdateLinearSolution(const State&);

    /// Computes residuals of the force array for the non-linear update
    HostView1D ComputeResiduals(HostView1D, std::function<HostView1D(size_t)>);

    /// Checks convergence of the non-linear solution based on the residuals
    bool CheckConvergence(HostView1D, HostView1D);

    /// Returns the flag to indicate if the latest non-linear update has converged
    inline bool IsConverged() const { return is_converged_; }

    /// Computes the iteration matrix for the non-linear update
    HostView2D ComputeIterationMatrix(HostView1D gen_coords, std::function<HostView2D(size_t)>);

private:
    const double kALPHA_F_;  //< Alpha_f coefficient of the generalized-alpha method
    const double kALPHA_M_;  //< Alpha_m coefficient of the generalized-alpha method
    const double kBETA_;     //< Beta coefficient of the generalized-alpha method
    const double kGAMMA_;    //< Gamma coefficient of the generalized-alpha method

    bool is_converged_;  //< Flag to indicate if the latest non-linear update has converged

    TimeStepper time_stepper_;  //< Time stepper object to perform the time integration
};

}  // namespace openturbine::rigid_pendulum
