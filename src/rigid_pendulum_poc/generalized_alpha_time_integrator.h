#pragma once

#include "src/rigid_pendulum_poc/state.h"
#include "src/rigid_pendulum_poc/time_integrator.h"
#include "src/rigid_pendulum_poc/time_stepper.h"

namespace openturbine::rigid_pendulum {

// TECHDEBT: Following is a hack to get around the fact that we don't have a way/scope to perform
// automatic differentiation yet to calculate the iteration matrix. This is a temporary solution
// until we implement something more robust.

/// Create an enum for the problem type
enum class ProblemType {
    kRigidBody = 0,     //< Arbitrary rigid body
    kHeavyTop = 1,      //< Heavy top problem
    kRigidPendulum = 2  //< Rigid pendulum problem
};

/// @brief A time integrator class based on the generalized-alpha method
class GeneralizedAlphaTimeIntegrator : public TimeIntegrator {
public:
    static constexpr double kTOLERANCE = 1e-6;

    GeneralizedAlphaTimeIntegrator(
        double alpha_f = 0.5, double alpha_m = 0.5, double beta = 0.25, double gamma = 0.5,
        TimeStepper time_stepper = TimeStepper(), ProblemType problem_type = ProblemType::kHeavyTop
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
        const State&, const MassMatrix&, const GeneralizedForces&, HostView1D,
        std::function<HostView2D(size_t)> iteration_matrix = create_identity_matrix,
        std::function<HostView1D(size_t)> residual_vector = create_identity_vector
    ) override;

    /*! @brief  Performs the SolveTimeStep() algorithm of the Lie group based generalized-alpha
     * method as described in the paper by Brüls and Cardona (2010)
     * https://doi.org/10.1115/1.4001370
     * @param   state Current state of the system at the beginning of the time step
     * @return  Updated state of the system at the end of the time step
     */
    std::tuple<State, HostView1D> AlphaStep(
        const State&, const MassMatrix&, const GeneralizedForces&, HostView1D,
        std::function<HostView2D(size_t)>, std::function<HostView1D(size_t)> residual_vector
    );

private:
    const double kALPHA_F_;  //< Alpha_f coefficient of the generalized-alpha method
    const double kALPHA_M_;  //< Alpha_m coefficient of the generalized-alpha method
    const double kBETA_;     //< Beta coefficient of the generalized-alpha method
    const double kGAMMA_;    //< Gamma coefficient of the generalized-alpha method

    bool is_converged_;         //< Flag to indicate if the latest non-linear update has converged
    TimeStepper time_stepper_;  //< Time stepper object to perform the time integration
    ProblemType problem_type_;  //< Type of the problem to be solved

    /// Computes the updated generalized coordinates based on the non-linear update
    HostView1D ComputeUpdatedGeneralizedCoordinates(HostView1D, HostView1D);

    /// Computes residuals of the force array for the non-linear update
    HostView1D ComputeResiduals(
        const MassMatrix&, const GeneralizedForces&, HostView1D, HostView1D, HostView1D,
        std::function<HostView1D(size_t)> vector
    );

    /// Checks convergence of the non-linear solution based on the residuals
    bool CheckConvergence(HostView1D);

    /// Returns the flag to indicate if the latest non-linear update has converged
    inline bool IsConverged() const { return is_converged_; }

    HostView2D ComputeIterationMatrix(
        const double&, const double&, const MassMatrix&, const GeneralizedForces&, HostView1D,
        HostView1D, HostView1D, std::function<HostView2D(size_t)> matrix
    );
};

}  // namespace openturbine::rigid_pendulum
