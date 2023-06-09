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

    /// Get the generalized coordinates dot
    inline HostView1D GetGeneralizedVelocity() const { return generalized_velocity_; }

    /// Get the generalized coordinates dot dot
    inline HostView1D GetGeneralizedAcceleration() const { return generalized_accelerations_; }

    /// Get the accelerations
    inline HostView1D GetAccelerations() const { return algorithmic_accelerations_; }

private:
    HostView1D generalized_coords_;
    HostView1D generalized_velocity_;
    HostView1D generalized_accelerations_;
    HostView1D algorithmic_accelerations_;
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

/*!
 * @brief Add two 1D kokkos views together using an overloaded addition operator
 * @param lhs The left hand side view
 * @param rhs The right hand side view
 * @return The sum of the two views as a new view
 */
HostView1D operator+(const HostView1D&, const HostView1D&);

/*!
 * @brief Overload the subtraction (i.e. -) operator to subtract two State objects
 *        from each other
 * @param lhs The left hand side state
 * @param rhs The right hand side state
 * @return The difference of the two states as a new state
 */
HostView1D operator-(const HostView1D&, const HostView1D&);

/*!
 * @brief Multiply a 1D kokkos view by a scalar using an overloaded multiplication operator
 * @param lhs The left hand side view
 * @param rhs The right hand side scalar
 * @return The product of the view and the scalar
 */
HostView1D operator*(const HostView1D&, double);
HostView1D operator*(double, const HostView1D&);

/// @brief A time integrator class based on the generalized-alpha method
class GeneralizedAlphaTimeIntegrator {
public:
    static constexpr double kALPHA_F = 0.5;
    static constexpr double kALPHA_M = 0.5;
    static constexpr double kBETA = 0.25;
    static constexpr double kGAMMA = 0.5;
    static constexpr size_t kMAX_ITERATIONS = 10;

    GeneralizedAlphaTimeIntegrator(
        double initial_time = 0., double time_step = 1., size_t number_of_steps = 1,
        State state = State(), State state_increment = State()
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

    /// Performs the time integration
    void Integrate();

    /// Returns the current state of the system
    inline State GetState() const { return state_; }

    /// Returns the incremental change in the state of the system
    inline State GetStateIncrement() const { return state_increment_; }

private:
    double initial_time_;     //< Initial time of the analysis
    double time_step_;        //< Time step of the analysis
    size_t number_of_steps_;  //< Number of time steps to perform
    double current_time_;     //< Current time of the analysis

    State state_;            //< Current state of the system
    State state_increment_;  //< Incremental change in the state of the system

    /*! @brief  Perform the alpha step of the generalized-alpha method as described
     *          in the paper by Arnold and Brüls (2007)
     *          https://link.springer.com/article/10.1007/s11044-007-9084-0
     * @param   state The current state of the system
     * @return  The updated state of the system
     */
    void AlphaStep();
};

}  // namespace openturbine::rigid_pendulum
