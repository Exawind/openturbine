#include "inflow.hpp"

#include <cmath>
#include <stdexcept>

namespace openturbine::interfaces::components {

std::array<double, 3> UniformFlowParameters::Velocity(const std::array<double, 3>& position) const {
    // Calculate horizontal velocity using power law wind shear
    // vh = v_ref * (z / z_ref)^alpha
    double vh = velocity_horizontal * std::pow(position[2] / height_reference, shear_vertical);

    // Apply horizontal direction to the velocity vector
    double sin_flow_angle = std::sin(flow_angle_horizontal);
    double cos_flow_angle = std::cos(flow_angle_horizontal);
    return {vh * cos_flow_angle, -vh * sin_flow_angle, 0.};
}

std::array<double, 3> UniformFlow::Velocity(
    [[maybe_unused]] double t, const std::array<double, 3>& position
) const {
    // If there is only one time point, use the uniform flow parameters
    if (time.size() == 1) {
        return data[0].Velocity(position);
    }

    // Multiple time points are not supported yet
    throw std::runtime_error("Time-dependent uniform flow not implemented yet");
}

Inflow Inflow::SteadyWind(double vh, double z_ref, double alpha, double flow_angle_h) {
    return Inflow{
        .type = InflowType::Uniform,
        .uniform_flow =
            UniformFlow{
                .time = {0.},
                .data = {UniformFlowParameters{
                    .velocity_horizontal = vh,
                    .height_reference = z_ref,
                    .shear_vertical = alpha,
                    .flow_angle_horizontal = flow_angle_h
                }}
            }
    };
}

std::array<double, 3> Inflow::Velocity(double t, const std::array<double, 3>& position) const {
    if (type == InflowType::Uniform) {
        return uniform_flow.Velocity(t, position);
    }
    throw std::runtime_error("Unknown inflow type");
}

}  // namespace openturbine::interfaces::components
