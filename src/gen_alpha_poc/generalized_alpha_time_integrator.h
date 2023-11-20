#pragma once

#include "src/gen_alpha_poc/linearization_parameters.h"
#include "src/gen_alpha_poc/state.h"
#include "src/gen_alpha_poc/time_integrator.h"
#include "src/gen_alpha_poc/time_stepper.h"

namespace openturbine::gen_alpha_solver {

/// @brief A time integrator class based on the generalized-alpha method
class GeneralizedAlphaTimeIntegrator : public TimeIntegrator {
public:
    static constexpr double kConvergenceTolerance = 1e-12;
    static constexpr size_t kNumberOfLieGroupComponents = 7;
    static constexpr size_t kNumberOfLieAlgebraComponents = 6;

    GeneralizedAlphaTimeIntegrator(
        double alpha_f = 0.5, double alpha_m = 0.5, double beta = 0.25, double gamma = 0.5,
        TimeStepper time_stepper = TimeStepper(), bool precondition = false
    );

    /// Returns the type of the time integrator
    inline TimeIntegratorType GetType() const override {
        return TimeIntegratorType::kGeneralizedAlpha;
    }

    /// Returns the alpha_f parameter
    inline double GetAlphaF() const { return kAlphaF_; }

    /// Returns the alpha_m parameter
    inline double GetAlphaM() const { return kAlphaM_; }

    /// Returns the beta parameter
    inline double GetBeta() const { return kBeta_; }

    /// Returns the gamma parameter
    inline double GetGamma() const { return kGamma_; }

    /// Returns a const reference to the time stepper
    inline const TimeStepper& GetTimeStepper() const { return time_stepper_; }

    /// Performs the time integration and returns a vector of States over the time steps
    virtual std::vector<State>
    Integrate(const State&, size_t, std::shared_ptr<LinearizationParameters>) override;

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
    void UpdateGeneralizedCoordinates(
        Kokkos::View<const double[kNumberOfLieGroupComponents]> gen_coords,
        Kokkos::View<const double[kNumberOfLieAlgebraComponents]> delta_gen_coords,
        Kokkos::View<double[kNumberOfLieGroupComponents]> gen_coords_next
    );

    /// Checks convergence of the non-linear solution based on the residuals
    bool CheckConvergence(const Kokkos::View<double*>);

    /// Returns the flag to indicate if the latest non-linear update has converged
    inline bool IsConverged() const { return is_converged_; }

private:
    const double kAlphaF_;  //< Alpha_f coefficient of the generalized-alpha method
    const double kAlphaM_;  //< Alpha_m coefficient of the generalized-alpha method
    const double kBeta_;    //< Beta coefficient of the generalized-alpha method
    const double kGamma_;   //< Gamma coefficient of the generalized-alpha method

    bool is_converged_;         //< Flag to indicate if the latest non-linear update has converged
    TimeStepper time_stepper_;  //< Time stepper object to perform the time integration
    bool is_preconditioned_;    //< Flag to indicate if the iteration matrix is preconditioned
};

}  // namespace openturbine::gen_alpha_solver
