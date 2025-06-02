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
    std::vector<BeamInput> blades;

    /// @brief Configuration for the tower
    BeamInput tower;

    /// @brief Yaw bearing inertia matrix (6x6)
    /// includes yaw bearing and nacelle mass with inertia about yaw bearing
    std::array<std::array<double, 6>, 6> yaw_bearing_inertia_matrix{
        {{0., 0., 0., 0., 0., 0.},
         {0., 0., 0., 0., 0., 0.},
         {0., 0., 0., 0., 0., 0.},
         {0., 0., 0., 0., 0., 0.},
         {0., 0., 0., 0., 0., 0.},
         {0., 0., 0., 0., 0., 1.}}
    };

    /// @brief Nacelle center of mass location relative to tower top
    std::array<double, 3> nacelle_mass_location{0., 0., 0.};

    /// @brief Distance from tower top to rotor apex (meters)
    double tower_top_to_rotor_apex{0.};

    /// @brief Distance from tower centerline to rotor apex (meters)
    double tower_axis_to_rotor_apex{0.};

    /// @brief Distance from rotor apex to hub center of mass (meters)
    double rotor_apex_to_hub{0.};

    /// @brief Shaft tilt angle (radians)
    double shaft_tilt_angle{0.};

    /// @brief Azimuth angle (radians)
    double azimuth_angle{0.};

    /// @brief Initial nacelle yaw angle (radians)
    double nacelle_yaw_angle{0.};

    /// @brief Initial blade pitch angle (radians)
    double blade_pitch_angle{0.};

    /// @brief Initial rotor speed (RPM)
    double rotor_speed{0.};

    /// @brief Hub diameter (meters)
    double hub_diameter{0.};

    /// @brief Define blade cone angle (radians)
    double cone_angle{0.};

    /// @brief Position of the tower base node in the global coordinate system
    std::array<double, 7> tower_base_position{0., 0., 0., 1., 0., 0., 0.};
};

}  // namespace openturbine::interfaces::components
