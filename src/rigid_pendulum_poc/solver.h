#pragma once

#include <Kokkos_Core.hpp>

namespace openturbine::rigid_pendulum {

using HostView1D = Kokkos::View<double*, Kokkos::HostSpace>;
using HostView2D = Kokkos::View<double**, Kokkos::HostSpace>;

/// @brief Solve a linear system of equations using LAPACKE's dgesv
/// @details This function solves a linear system of equations using LAPACKE's
///     dgesv function. The system is of the form Ax = b, where A is a square matrix
///     (nxn) of coefficients, x is a vector (nx1) of unknowns, and b is a vector
///     (nx1) of right-hand side values. The solution is stored in the vector b.
/// @param system A matrix of coefficients
/// @param solution A vector of right-hand side values
void solve_linear_system(HostView2D, HostView1D);

/// @brief A class to store and manage the states of a dynamic system
// TODO: Refactor this class to avoid/minimize expensive copies
class State {
public:
    State();
    State(HostView1D, HostView1D, HostView1D, HostView1D);

    /// Get the generalized coordinates
    inline HostView1D GetGeneralizedCoordinates() const { return generalized_coords_; }

    /// Get the first time derivative of the generalized coordinates
    inline HostView1D GetGeneralizedVelocity() const { return generalized_velocity_; }

    /// Get the second time derivative of the generalized coordinates
    inline HostView1D GetGeneralizedAcceleration() const { return generalized_acceleration_; }

    /// Get the algorithmic accelerations (different than the generalized accelerations)
    inline HostView1D GetAlgorithmicAcceleration() const { return algorithmic_acceleration_; }

private:
    HostView1D generalized_coords_;
    HostView1D generalized_velocity_;
    HostView1D generalized_acceleration_;
    HostView1D algorithmic_acceleration_;
};

HostView2D create_identity_matrix(size_t size);
HostView1D create_identity_vector(size_t size);

/// @brief A time integrator class based on the generalized-alpha method
class GeneralizedAlphaTimeIntegrator {
public:
    static constexpr double kTOLERANCE = 1e-6;

    GeneralizedAlphaTimeIntegrator(
        double initial_time = 0., double time_step = 1., size_t n_steps = 1, double alpha_f = 0.5,
        double alpha_m = 0.5, double beta = 0.25, double gamma = 0.5, size_t max_iterations = 10
    );

    /// Returns the initial time of the analysis
    inline double GetInitialTime() const { return initial_time_; }

    /// Returns the current time of the analysis
    inline double GetCurrentTime() const { return current_time_; }

    /// Returns the time step of the analysis
    inline double GetTimeStep() const { return time_step_; }

    /// Advances the current analysis time by one time step
    inline void AdvanceTimeStep() { current_time_ += time_step_; }

    /// Returns the number of analysis time steps
    inline int GetNumberOfSteps() const { return n_steps_; }

    /// Returns the alpha_f parameter
    inline double GetAlphaF() const { return kALPHA_F_; }

    /// Returns the alpha_m parameter
    inline double GetAlphaM() const { return kALPHA_M_; }

    /// Returns the beta parameter
    inline double GetBeta() const { return kBETA_; }

    /// Returns the gamma parameter
    inline double GetGamma() const { return kGAMMA_; }

    /// Returns the maximum number of iterations
    inline size_t GetMaxIterations() const { return kMAX_ITERATIONS_; }

    /// Performs the time integration and returns a vector of States over the time steps
    std::vector<State> Integrate(
        const State&, std::function<HostView2D(size_t)> iteration_matrix = create_identity_matrix,
        std::function<HostView1D(size_t)> residual_vector = create_identity_vector
    );

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

    /// Returns the number of iterations performed in the latest non-linear update
    inline size_t GetNumberOfIterations() const { return n_iterations_; }

    /// Returns the total number of iterations performed to complete the analysis
    inline size_t GetTotalNumberOfIterations() const { return total_n_iterations_; }

    /// Returns the flag to indicate if the latest non-linear update has converged
    inline bool IsConverged() const { return is_converged_; }

    /// Computes the iteration matrix for the non-linear update
    HostView2D ComputeIterationMatrix(HostView1D gen_coords, std::function<HostView2D(size_t)>);

private:
    double initial_time_;        //< Initial time of the analysis
    double time_step_;           //< Time step of the analysis
    size_t n_steps_;             //< Number of time steps to perform
    double current_time_;        //< Current time of the analysis
    size_t n_iterations_;        //< Number of iterations performed in the latest non-linear update
    size_t total_n_iterations_;  //< Total number of non-linear iterations performed to
                                 // complete the analysis
    bool is_converged_;          //< Flag to indicate if the latest non-linear update has converged

    const double kALPHA_F_;         //< Alpha_f coefficient of the generalized-alpha method
    const double kALPHA_M_;         //< Alpha_m coefficient of the generalized-alpha method
    const double kBETA_;            //< Beta coefficient of the generalized-alpha method
    const double kGAMMA_;           //< Gamma coefficient of the generalized-alpha method
    const size_t kMAX_ITERATIONS_;  //< Maximum number of iterations for the non-linear update
};

}  // namespace openturbine::rigid_pendulum
