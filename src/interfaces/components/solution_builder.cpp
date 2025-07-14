#include "solution_builder.hpp"

namespace openturbine::interfaces::components {
SolutionBuilder& SolutionBuilder::EnableStaticSolve() {
    input.dynamic_solve = false;
    return *this;
}

SolutionBuilder& SolutionBuilder::EnableDynamicSolve() {
    input.dynamic_solve = true;
    return *this;
}

SolutionBuilder& SolutionBuilder::SetGravity(const std::array<double, 3>& gravity) {
    input.gravity = gravity;
    return *this;
}

SolutionBuilder& SolutionBuilder::SetRelativeErrorTolerance(double rtol) {
    input.relative_error_tolerance = rtol;
    return *this;
}

SolutionBuilder& SolutionBuilder::SetAbsoluteErrorTolerance(double atol) {
    input.absolute_error_tolerance = atol;
    return *this;
}

SolutionBuilder& SolutionBuilder::SetMaximumNonlinearIterations(size_t max_iter) {
    input.max_iter = max_iter;
    return *this;
}

SolutionBuilder& SolutionBuilder::SetTimeStep(double time_step) {
    if (time_step < 0.) {
        throw std::out_of_range("time_step must be positive");
    }
    input.time_step = time_step;
    return *this;
}

SolutionBuilder& SolutionBuilder::SetDampingFactor(double rho_inf) {
    if (rho_inf < 0. || rho_inf > 1.) {
        throw std::out_of_range("rho_inf must be in range [0., 1.]");
    }
    input.rho_inf = rho_inf;
    return *this;
}

SolutionBuilder& SolutionBuilder::SetOutputFile(const std::string& output_file_path) {
    input.output_file_path = output_file_path;
    return *this;
}

const SolutionInput& SolutionBuilder::Input() const {
    return this->input;
}

}  // namespace openturbine::interfaces::components
