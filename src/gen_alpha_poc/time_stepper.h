#pragma once

#include <cstddef>

namespace openturbine::gen_alpha_solver {

/// @brief A class to store and manage the states of a dynamic system
class TimeStepper {
public:
    TimeStepper(
        double initial_time = 0., double time_step = 1., size_t n_steps = 1,
        size_t max_iterations = 10
    );

    /// Returns the initial time of the analysis
    inline double GetInitialTime() const { return initial_time_; }

    /// Returns the current time of the analysis
    inline double GetCurrentTime() const { return current_time_; }

    /// Returns the time step of the analysis
    inline double GetTimeStep() const { return time_step_; }

    /// Advances the current analysis time by one time step
    inline void AdvanceTimeStep() { current_time_ += time_step_; }

    /// Returns the number of analysis time steps to perform
    inline size_t GetNumberOfSteps() const { return n_steps_; }

    /// Returns the number of iterations performed in the latest time step
    inline size_t GetNumberOfIterations() const { return n_iterations_; }

    /// Setter for number of iterations
    inline void SetNumberOfIterations(size_t n_iterations) { n_iterations_ = n_iterations; }

    /// Increments the number of iterations by one
    inline void IncrementNumberOfIterations() { n_iterations_++; }

    /// Returns the total number of iterations performed thus far (across all time steps)
    inline size_t GetTotalNumberOfIterations() const { return total_n_iterations_; }

    /// Increments the total number of iterations by the number of iterations performed
    inline void IncrementTotalNumberOfIterations(size_t n) { total_n_iterations_ += n; }

    /// Returns the maximum number of iterations for the non-linear update
    inline size_t GetMaximumNumberOfIterations() const { return kMaxIterations_; }

private:
    double initial_time_;          //< Initial time of the analysis
    double time_step_;             //< Time step (delta t) of the analysis
    size_t n_steps_;               //< Number of time steps to perform in the analysis
    double current_time_;          //< Current time of the analysis
    size_t n_iterations_;          //< Number of iterations performed in the latest non-linear update
    size_t total_n_iterations_;    //< Total number of non-linear iterations performed to
                                   // complete the analysis
    const size_t kMaxIterations_;  //< Maximum number of iterations permitted for each
                                   // non-linear update
};

}  // namespace openturbine::gen_alpha_solver
