#pragma once

#include "floating_platform_input.hpp"
#include "interface.hpp"
#include "interface_input.hpp"
#include "mooring_line_input.hpp"

namespace openturbine::cfd {

struct InterfaceBuilder {
    InterfaceBuilder& SetGravity(const std::array<double, 3>& gravity) {
        interface_in.gravity = gravity;
        return *this;
    }

    InterfaceBuilder& SetMaximumNonlinearIterations(size_t max_iter) {
        interface_in.max_iter = max_iter;
        return *this;
    }

    InterfaceBuilder& SetTimeStep(double time_step) {
        interface_in.time_step = time_step;
        return *this;
    }

    InterfaceBuilder& SetDampingFactor(double rho_inf) {
        interface_in.rho_inf = rho_inf;
        return *this;
    }

    InterfaceBuilder& EnableFloatingPlatform(bool enable) {
        interface_in.turbine.floating_platform.enable = enable;
        return *this;
    }

    InterfaceBuilder& SetFloatingPlatformPosition(const std::array<double, 7>& position) {
        interface_in.turbine.floating_platform.position = position;
        return *this;
    }

    InterfaceBuilder& SetFloatingPlatformVelocity(const std::array<double, 6>& velocity) {
        interface_in.turbine.floating_platform.velocity = velocity;
        return *this;
    }

    InterfaceBuilder& SetFloatingPlatformAcceleration(const std::array<double, 6>& acceleration) {
        interface_in.turbine.floating_platform.acceleration = acceleration;
        return *this;
    }

    InterfaceBuilder& SetFloatingPlatformMassMatrix(
        const std::array<std::array<double, 6>, 6>& mass_matrix
    ) {
        interface_in.turbine.floating_platform.mass_matrix = mass_matrix;
        return *this;
    }

    InterfaceBuilder& SetNumberOfMooringLines(size_t number_of_mooring_lines) {
        interface_in.turbine.floating_platform.mooring_lines.resize(number_of_mooring_lines);
        return *this;
    }

    InterfaceBuilder& SetMooringLineStiffness(size_t line_number, double stiffness) {
        interface_in.turbine.floating_platform.mooring_lines[line_number].stiffness = stiffness;
        return *this;
    }

    InterfaceBuilder& SetMooringLineUndeformedLength(size_t line_number, double length) {
        interface_in.turbine.floating_platform.mooring_lines[line_number].undeformed_length = length;
        return *this;
    }

    InterfaceBuilder& SetMooringLineFairleadPosition(
        size_t line_number, const std::array<double, 3>& position
    ) {
        interface_in.turbine.floating_platform.mooring_lines[line_number].fairlead_position =
            position;
        return *this;
    }

    InterfaceBuilder& SetMooringLineFairleadVelocity(
        size_t line_number, const std::array<double, 3>& velocity
    ) {
        interface_in.turbine.floating_platform.mooring_lines[line_number].fairlead_velocity =
            velocity;
        return *this;
    }

    InterfaceBuilder& SetMooringLineFairleadAcceleration(
        size_t line_number, const std::array<double, 3>& acceleration
    ) {
        interface_in.turbine.floating_platform.mooring_lines[line_number].fairlead_acceleration =
            acceleration;
        return *this;
    }

    InterfaceBuilder& SetMooringLineAnchorPosition(
        size_t line_number, const std::array<double, 3>& position
    ) {
        interface_in.turbine.floating_platform.mooring_lines[line_number].anchor_position = position;
        return *this;
    }

    InterfaceBuilder& SetMooringLineAnchorVelocity(
        size_t line_number, const std::array<double, 3>& velocity
    ) {
        interface_in.turbine.floating_platform.mooring_lines[line_number].anchor_velocity = velocity;
        return *this;
    }

    InterfaceBuilder& SetMooringLineAnchorAcceleration(
        size_t line_number, const std::array<double, 3>& acceleration
    ) {
        interface_in.turbine.floating_platform.mooring_lines[line_number].anchor_acceleration =
            acceleration;
        return *this;
    }

    InterfaceBuilder& SetTurbine(const TurbineInput& turbine_in) {
        interface_in.turbine = turbine_in;
        return *this;
    }

    [[nodiscard]] Interface Build() const { return Interface(interface_in); }

private:
    InterfaceInput interface_in;
};

}  // namespace openturbine::cfd
