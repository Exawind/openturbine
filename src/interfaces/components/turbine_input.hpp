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
    std::array<std::array<double, 6>, 6> nacelle_mass_matrix{
        {{0., 0., 0., 0., 0., 0.},
         {0., 0., 0., 0., 0., 0.},
         {0., 0., 0., 0., 0., 0.},
         {0., 0., 0., 0., 0., 0.},
         {0., 0., 0., 0., 0., 0.},
         {0., 0., 0., 0., 0., 0.}}
    };

    /// @brief Distance from tower top to rotor apex (meters)
    double tower_top_to_hub{0.};

    /// @brief Distance from tower centerline to rotor apex (meters)
    double tower_axis_to_hub{0.};

    /// @brief Shaft tilt angle (degrees)
    double shaft_tilt_angle{0.};

    /// @brief Azimuth angle (degrees)
    double azimuth_angle{0.};

    /// @brief Initial nacelle yaw angle (degrees)
    double nacelle_yaw_angle{0.};

    /// @brief Initial blade pitch angle (degrees)
    double blade_pitch_angle{0.};

    /// @brief Initial rotor speed (RPM)
    double rotor_speed{0.};

    /// @brief Controller enabled flag
    bool controller_enabled{false};
};

}  // namespace openturbine::interfaces::components
