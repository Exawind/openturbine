#include "src/gen_alpha_poc/time_stepper.h"

#include <stdexcept>

namespace openturbine::gen_alpha_solver {

TimeStepper::TimeStepper(
    double initial_time, double time_step, size_t n_steps, size_t max_iterations
)
    : initial_time_(initial_time),
      time_step_(time_step),
      n_steps_(n_steps),
      kMaxIterations_(max_iterations) {
    this->current_time_ = initial_time;
    this->n_iterations_ = 0;
    this->total_n_iterations_ = 0;

    if (this->kMaxIterations_ < 1) {
        throw std::invalid_argument("Invalid value for max_iterations");
    }
}

}  // namespace openturbine::gen_alpha_solver
