#pragma once

#include "floating_platform_input.hpp"
#include "interface.hpp"
#include "interface_input.hpp"
#include "mooring_line_input.hpp"

namespace openturbine::cfd {

struct InterfaceBuilder {
    InterfaceBuilder& SetGravity(const std::array<double, 3>& gravity);

    InterfaceBuilder& SetMaximumNonlinearIterations(size_t max_iter);

    InterfaceBuilder& SetTimeStep(double time_step);

    InterfaceBuilder& SetDampingFactor(double rho_inf);

    InterfaceBuilder& EnableFloatingPlatform(bool enable);

    InterfaceBuilder& SetFloatingPlatformPosition(const std::array<double, 7>& p);

    InterfaceBuilder& SetFloatingPlatformVelocity(const std::array<double, 6>& v);

    InterfaceBuilder& SetFloatingPlatformAcceleration(const std::array<double, 6>& a);

    InterfaceBuilder& SetFloatingPlatformMassMatrix(
        const std::array<std::array<double, 6>, 6>& mass_matrix
    );

    InterfaceBuilder& SetNumberOfMooringLines(size_t number);

    InterfaceBuilder& SetMooringLineStiffness(size_t line, double stiffness);

    InterfaceBuilder& SetMooringLineUndeformedLength(size_t line, double length);

    InterfaceBuilder& SetMooringLineFairleadPosition(size_t line, const std::array<double, 3>& p);

    InterfaceBuilder& SetMooringLineAnchorPosition(size_t line, const std::array<double, 3>& p);

    InterfaceBuilder& SetMooringLineAnchorVelocity(size_t line, const std::array<double, 3>& v);

    InterfaceBuilder& SetMooringLineAnchorAcceleration(size_t line, const std::array<double, 3>& a);

    InterfaceBuilder& SetOutputFile(const std::string& path);

    [[nodiscard]] Interface Build() const;

private:
    InterfaceInput interface_input;
};

}  // namespace openturbine::cfd
