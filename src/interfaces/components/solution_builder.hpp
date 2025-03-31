#pragma once

#include "model/model.hpp"
#include "solution_input.hpp"
#include "solver/solver.hpp"

namespace openturbine::interfaces::components {

struct SolutionBuilder {
    SolutionBuilder& EnableStaticSolve() {
        input.dynamic_solve = false;
        return *this;
    }

    SolutionBuilder& EnableDynamicSolve() {
        input.dynamic_solve = true;
        return *this;
    }

    SolutionBuilder& SetGravity(const std::array<double, 3>& gravity) {
        input.gravity = gravity;
        return *this;
    }

    SolutionBuilder& SetRelativeErrorTolerance(double rtol) {
        input.relative_error_tolerance = rtol;
        return *this;
    }

    SolutionBuilder& SetAbsoluteErrorTolerance(double atol) {
        input.absolute_error_tolerance = atol;
        return *this;
    }

    SolutionBuilder& SetMaximumNonlinearIterations(size_t max_iter) {
        input.max_iter = max_iter;
        return *this;
    }

    SolutionBuilder& SetTimeStep(double time_step) {
        if (time_step < 0.) {
            throw std::out_of_range("time_step must be positive");
        }
        input.time_step = time_step;
        return *this;
    }

    SolutionBuilder& SetDampingFactor(double rho_inf) {
        if (rho_inf < 0. || rho_inf > 1.) {
            throw std::out_of_range("rho_inf must be in range [0., 1.]");
        }
        input.rho_inf = rho_inf;
        return *this;
    }

    SolutionBuilder& SetOutputFile(const std::string& output_file_path) {
        input.output_file_path = output_file_path;
        return *this;
    }

    SolutionBuilder& SetVTKOutputPath(const std::string& vtk_output_path) {
        input.vtk_output_path = vtk_output_path;
        return *this;
    }

    [[nodiscard]] const SolutionInput& Input() const { return this->input; }

private:
    SolutionInput input;
};

}  // namespace openturbine::interfaces::components
