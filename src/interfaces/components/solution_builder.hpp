#pragma once

#include "solution_input.hpp"

namespace openturbine::interfaces::components {

struct SolutionBuilder {
    SolutionBuilder& EnableStaticSolve();

    SolutionBuilder& EnableDynamicSolve();

    SolutionBuilder& SetGravity(const std::array<double, 3>& gravity);

    SolutionBuilder& SetRelativeErrorTolerance(double rtol);

    SolutionBuilder& SetAbsoluteErrorTolerance(double atol);

    SolutionBuilder& SetMaximumNonlinearIterations(size_t max_iter);

    SolutionBuilder& SetTimeStep(double time_step);

    SolutionBuilder& SetDampingFactor(double rho_inf);

    SolutionBuilder& SetOutputFile(const std::string& output_file_path);

    [[nodiscard]] const SolutionInput& Input() const;

private:
    SolutionInput input;
};

}  // namespace openturbine::interfaces::components
