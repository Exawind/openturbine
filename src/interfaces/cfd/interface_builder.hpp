#pragma once

#include "floating_platform_input.hpp"
#include "interface.hpp"
#include "interface_input.hpp"
#include "mooring_line_input.hpp"

namespace openturbine::cfd {
struct InterfaceBuilder {
    struct FloatingPlatformBuilder {
        FloatingPlatformBuilder& SetPosition(const std::array<double, 7>& p) {
            i_builder->interface_input.turbine.floating_platform.position = p;
            return *this;
        }

        FloatingPlatformBuilder& SetVelocity(const std::array<double, 6>& v) {
            i_builder->interface_input.turbine.floating_platform.velocity = v;
            return *this;
        }

        FloatingPlatformBuilder& SetAcceleration(const std::array<double, 6>& a) {
            i_builder->interface_input.turbine.floating_platform.acceleration = a;
            return *this;
        }

        FloatingPlatformBuilder& SetMassMatrix(
            const std::array<std::array<double, 6>, 6>& mass_matrix
        ) {
            i_builder->interface_input.turbine.floating_platform.mass_matrix = mass_matrix;
            return *this;
        }

        [[nodiscard]] InterfaceBuilder& EndFloatingPlatform() const {
            i_builder->interface_input.turbine.floating_platform.enable = true;
            return *i_builder;
        }

        InterfaceBuilder* i_builder;
    };

    struct MooringLineBuilder {
        MooringLineBuilder& SetStiffness(double stiffness) {
            i_builder->interface_input.turbine.floating_platform.mooring_lines.back().stiffness =
                stiffness;
            return *this;
        }

        MooringLineBuilder& SetUndeformedLength(double length) {
            i_builder->interface_input.turbine.floating_platform.mooring_lines.back()
                .undeformed_length = length;
            return *this;
        }

        MooringLineBuilder& SetFairleadPosition(const std::array<double, 3>& p) {
            i_builder->interface_input.turbine.floating_platform.mooring_lines.back()
                .fairlead_position = p;
            return *this;
        }

        MooringLineBuilder& SetAnchorPosition(const std::array<double, 3>& p) {
            i_builder->interface_input.turbine.floating_platform.mooring_lines.back()
                .anchor_position = p;
            return *this;
        }

        MooringLineBuilder& SetAnchorVelocity(const std::array<double, 3>& v) {
            i_builder->interface_input.turbine.floating_platform.mooring_lines.back()
                .anchor_velocity = v;
            return *this;
        }

        MooringLineBuilder& SetAnchorAcceleration(const std::array<double, 3>& a) {
            i_builder->interface_input.turbine.floating_platform.mooring_lines.back()
                .anchor_acceleration = a;
            return *this;
        }

        [[nodiscard]] InterfaceBuilder& EndMooringLine() const { return *i_builder; }

        InterfaceBuilder* i_builder;
    };

    InterfaceBuilder() : interface_input{}, fp_builder{this}, ml_builder{this} {}

    InterfaceBuilder& SetGravity(const std::array<double, 3>& gravity) {
        interface_input.gravity = gravity;
        return *this;
    }

    InterfaceBuilder& SetMaximumNonlinearIterations(size_t max_iter) {
        interface_input.max_iter = max_iter;
        return *this;
    }

    InterfaceBuilder& SetTimeStep(double time_step) {
        interface_input.time_step = time_step;
        return *this;
    }

    FloatingPlatformBuilder& StartFloatingPlatform() { return fp_builder; }

    MooringLineBuilder& AddMooringLine() {
        interface_input.turbine.floating_platform.mooring_lines.push_back({});
        return ml_builder;
    }

    InterfaceBuilder& SetDampingFactor(double rho_inf) {
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
        interface_input.turbine.floating_platform.mooring_lines[line].stiffness = stiffness;
        return *this;
    }

    InterfaceBuilder& SetMooringLineUndeformedLength(size_t line, double length) {
        interface_input.turbine.floating_platform.mooring_lines[line].undeformed_length = length;
        return *this;
    }

    InterfaceBuilder& SetMooringLineFairleadPosition(size_t line, const std::array<double, 3>& p) {
        interface_input.turbine.floating_platform.mooring_lines[line].fairlead_position = p;
        return *this;
    }

    InterfaceBuilder& SetMooringLineAnchorPosition(size_t line, const std::array<double, 3>& p) {
        interface_input.turbine.floating_platform.mooring_lines[line].anchor_position = p;
        return *this;
    }

    InterfaceBuilder& SetMooringLineAnchorVelocity(size_t line, const std::array<double, 3>& v) {
        interface_input.turbine.floating_platform.mooring_lines[line].anchor_velocity = v;
        return *this;
    }

    InterfaceBuilder& SetMooringLineAnchorAcceleration(size_t line, const std::array<double, 3>& a) {
        interface_input.turbine.floating_platform.mooring_lines[line].anchor_acceleration = a;
        return *this;
    }

    [[nodiscard]] Interface Build() const { return Interface(interface_input); }

    friend FloatingPlatformBuilder;
    friend MooringLineBuilder;

protected:
    InterfaceInput interface_input;

private:
    FloatingPlatformBuilder fp_builder;
    MooringLineBuilder ml_builder;
};

}  // namespace openturbine::cfd
