#pragma once

#include "types.hpp"

namespace openturbine::interfaces::components {

/**
 * @brief Complete input specification for a nacelle
 *
 * Defines the input configuration for a turbine nacelle
 */
struct NacelleInput {
    /// @brief Tower top position and orientation
    std::array<double, 7> tower_top_position{0., 0., 0., 1., 0., 0., 0.};

    /// @brief Nacelle center of mass position relative to tower top node
    std::array<double, 3> cm_position{0., 0., 0.};

    /// @brief Shaft base position relative to tower top node
    std::array<double, 3> shaft_base_position{0., 0., 0.};
};

}  // namespace openturbine::interfaces::components
