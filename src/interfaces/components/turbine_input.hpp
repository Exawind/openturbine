#pragma once

#include "interfaces/components/beam_input.hpp"
#include "types.hpp"

namespace openturbine::interfaces::components {

/**
 * @brief Complete input specification for a turbine
 *
 * Defines the input configuration for a turbine including blades, tower, hub, and nacelle.
 */
struct TurbineInput {
    /// @brief Configuration for the blades
    std::vector<BeamInput> blade_inputs;

    /// @brief Configuration for the tower
    BeamInput tower_input;

    /// @brief Nacelle center of mass position relative to tower top node
    std::array<double, 3> nacelle_mass_position{0., 0., 0.};

    /// @brief Nacelle mass matrix (6x6)
    std::array<std::array<double, 6>, 6> nacelle_mass_matrix{0., 0., 0., 0., 0., 0., 0., 0., 0.};

    /// @brief Shaft base position relative to tower top node
    std::array<double, 3> shaft_base_position{0., 0., 0.};

    /// @brief Shaft length (meters) from shaft base to hub
    double shaft_length{0.};

    /// @brief Shaft tilt angle (degrees)
    double shaft_tilt_angle{0.};

    /// @brief Azimuth angle (degrees)
    double azimuth_angle{0.};
};

}  // namespace openturbine::interfaces::components
