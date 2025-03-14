#pragma once

#include "floating_platform_input.hpp"
#include "interface.hpp"
#include "interface_input.hpp"
#include "mooring_line_input.hpp"

namespace openturbine::cfd {

struct InterfaceBuilder {
    InterfaceBuilder& SetGravity(const std::array<double, 3>& gravity) {
        interface_input.gravity = gravity;
        return *this;
    }

    InterfaceBuilder& SetMaximumNonlinearIterations(size_t max_iter) {
        interface_input.max_iter = max_iter;
        return *this;
    }

    InterfaceBuilder& SetTimeStep(double time_step) {
        if (time_step < 0.) {
            throw std::out_of_range("time_step must be positive");
        }
        interface_input.time_step = time_step;
        return *this;
    }

    InterfaceBuilder& SetDampingFactor(double rho_inf) {
        if (rho_inf < 0. || rho_inf > 1.) {
            throw std::out_of_range("rho_inf must be in range [0., 1.]");
        }
        interface_input.rho_inf = rho_inf;
        return *this;
    }

    InterfaceBuilder& EnableFloatingPlatform(bool enable) {
        interface_input.turbine.floating_platform.enable = enable;
        return *this;
    }

    InterfaceBuilder& SetFloatingPlatformPosition(const std::array<double, 7>& p) {
        interface_input.turbine.floating_platform.position = p;
        return *this;
    }

    InterfaceBuilder& SetFloatingPlatformVelocity(const std::array<double, 6>& v) {
        interface_input.turbine.floating_platform.velocity = v;
        return *this;
    }

    InterfaceBuilder& SetFloatingPlatformAcceleration(const std::array<double, 6>& a) {
        interface_input.turbine.floating_platform.acceleration = a;
        return *this;
    }

    InterfaceBuilder& SetFloatingPlatformMassMatrix(
        const std::array<std::array<double, 6>, 6>& mass_matrix
    ) {
        interface_input.turbine.floating_platform.mass_matrix = mass_matrix;
        return *this;
    }

    InterfaceBuilder& SetNumberOfMooringLines(size_t number) {
        interface_input.turbine.floating_platform.mooring_lines.resize(number);
        return *this;
    }

    InterfaceBuilder& SetMooringLineStiffness(size_t line, double stiffness) {
        if (stiffness < 0.) {
            throw std::out_of_range("stiffness must be positive");
        }
        interface_input.turbine.floating_platform.mooring_lines.at(line).stiffness = stiffness;
        return *this;
    }

    InterfaceBuilder& SetMooringLineUndeformedLength(size_t line, double length) {
        if (length < 0.) {
            throw std::out_of_range("undeformed length must be positive");
        }
        interface_input.turbine.floating_platform.mooring_lines.at(line).undeformed_length = length;
        return *this;
    }

    InterfaceBuilder& SetMooringLineFairleadPosition(size_t line, const std::array<double, 3>& p) {
        interface_input.turbine.floating_platform.mooring_lines.at(line).fairlead_position = p;
        return *this;
    }

    InterfaceBuilder& SetMooringLineAnchorPosition(size_t line, const std::array<double, 3>& p) {
        interface_input.turbine.floating_platform.mooring_lines.at(line).anchor_position = p;
        return *this;
    }

    InterfaceBuilder& SetMooringLineAnchorVelocity(size_t line, const std::array<double, 3>& v) {
        interface_input.turbine.floating_platform.mooring_lines.at(line).anchor_velocity = v;
        return *this;
    }

    InterfaceBuilder& SetMooringLineAnchorAcceleration(size_t line, const std::array<double, 3>& a) {
        interface_input.turbine.floating_platform.mooring_lines.at(line).anchor_acceleration = a;
        return *this;
    }

    InterfaceBuilder& SetOutputFile(const std::string& path) {
        interface_input.output_file = path;
        return *this;
    }

    [[nodiscard]] Interface Build() const { return Interface(interface_input); }

private:
    InterfaceInput interface_input;
};

}  // namespace openturbine::cfd
