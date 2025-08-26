#pragma once

#include <array>
#include <cstdint>
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
    [[nodiscard]] std::array<double, 3> Velocity(const std::array<double, 3>& position) const;
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
    [[nodiscard]] std::array<double, 3> Velocity(double t, const std::array<double, 3>& position)
        const;
};

/**
 * @brief Type of inflow model
 */
enum class InflowType : std::uint8_t {
    Uniform = 1,  ///< Uniform flow
};

/**
 * @brief Wind inflow model for turbine simulations
 */
struct Inflow {
    InflowType type;           ///< Type of inflow model
    UniformFlow uniform_flow;  ///< Uniform flow parameters

    /// Default constructor
    Inflow() = default;

    /// Constructor with type and uniform flow parameters
    Inflow(InflowType type, UniformFlow uniform_flow)
        : type(type), uniform_flow(std::move(uniform_flow)) {}

    /**
     * @brief Creates a steady wind inflow
     * @param vh Horizontal inflow velocity (m/s)
     * @param z_ref Reference height (m)
     * @param alpha Vertical shear exponent
     * @param angle_h Flow angle relative to x axis (radians)
     */
    static Inflow SteadyWind(double vh, double z_ref, double alpha, double angle_h);

    /**
     * @brief Calculates velocity vector at a given time and position
     * @param t Time
     * @param position 3D position [x, y, z]
     * @return 3D velocity vector [vx, vy, vz]
     */
    [[nodiscard]] std::array<double, 3> Velocity(double t, const std::array<double, 3>& position)
        const;
};

}  // namespace openturbine::interfaces::components
