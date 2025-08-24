#pragma once

#include <array>
#include <vector>

namespace openturbine::interfaces::components {

/**
 * @brief Parameters defining uniform flow characteristics
 */
struct UniformFlowParameters {
    double velocity_horizontal;    ///< Horizontal inflow velocity (m/s)
    double height_reference;       ///< Reference height (m)
    double shear_vertical;         ///< Vertical shear exponent
    double flow_angle_horizontal;  ///< Flow angle relative to x axis (radians)

    /**
     * @brief Calculates velocity vector at a given position
     * @param position 3D position [x, y, z]
     * @return 3D velocity vector [vx, vy, vz]
     */
    std::array<double, 3> Velocity(const std::array<double, 3>& position) const;
};

/**
 * @brief Uniform flow with time-dependent parameters
 */
struct UniformFlow {
    std::vector<double> time;  ///< Time vector for uniform flow parameters
    std::vector<UniformFlowParameters> data;

    /**
     * @brief Calculates velocity vector at a given time and position
     * @param t Time
     * @param position 3D position [x, y, z]
     * @return 3D velocity vector [vx, vy, vz]
     */
    std::array<double, 3> Velocity(double t, const std::array<double, 3>& position) const;
};

/**
 * @brief Type of inflow model
 */
enum class InflowType {
    Uniform = 1,  ///< Uniform flow
};

/**
 * @brief Wind inflow model for turbine simulations
 */
struct Inflow {
    InflowType type;           ///< Type of inflow model
    UniformFlow uniform_flow;  ///< Uniform flow parameters

    /**
     * @brief Creates a steady wind inflow
     * @param velocity_horizontal Horizontal inflow velocity (m/s)
     * @param height_reference Reference height (m)
     * @param shear_vertical Vertical shear exponent
     * @param flow_angle_horizontal Flow angle relative to x axis (radians)
     */
    static Inflow SteadyWind(double vh, double z_ref, double alpha, double flow_angle_h);

    /**
     * @brief Calculates velocity vector at a given time and position
     * @param t Time
     * @param position 3D position [x, y, z]
     * @return 3D velocity vector [vx, vy, vz]
     */
    std::array<double, 3> Velocity(double t, const std::array<double, 3>& position) const;
};

}  // namespace openturbine::interfaces::components
