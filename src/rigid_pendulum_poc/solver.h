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

/*!
 * @brief Overload the addition (i.e. +) operator to add two State objects together
 * @param lhs The left hand side state
 * @param rhs The right hand side state
 * @return The sum of the two states as a new state
 */
State operator+(const State&, const State&);

/*! @brief Overload the addition assignment (i.e. +=) operator to add two State
 *         objects together
 *  @param lhs The left hand side state
 *  @param rhs The right hand side state
 *  @return The sum of the two states, assigned to the left hand side state
 */
State operator+=(State&, const State&);

/// @brief A time integrator class based on the generalized-alpha method
class GeneralizedAlphaTimeIntegrator {
public:
    static constexpr double kALPHA_F = 0.5;
    static constexpr double kALPHA_M = 0.5;
    static constexpr double kBETA = 0.25;
    static constexpr double kGAMMA = 0.5;
    static constexpr size_t kMAX_ITERATIONS = 10;

    GeneralizedAlphaTimeIntegrator(
        double initial_time = 0., double time_step = 1., size_t number_of_steps = 1
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
    inline int GetNumberOfSteps() const { return number_of_steps_; }

    /// Performs the time integration and returns a vector of States over the time steps
    std::vector<State> Integrate(const State&);

private:
    double initial_time_;     //< Initial time of the analysis
    double time_step_;        //< Time step of the analysis
    size_t number_of_steps_;  //< Number of time steps to perform
    double current_time_;     //< Current time of the analysis

    /*! @brief  Perform the alpha step of the generalized-alpha method as described
     *          in the paper by Arnold and Brüls (2007)
     *          https://link.springer.com/article/10.1007/s11044-007-9084-0
     * @param   state Current state of the system at the beginning of the time step
     * @return  Updated state of the system at the end of the time step
     */
    State AlphaStep(const State&);

    /// Perform the linear part of the generalized-alpha method
    State UpdateLinearSolution(const State&);
};

}  // namespace openturbine::rigid_pendulum
