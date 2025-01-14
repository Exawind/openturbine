#pragma once

#include <array>
#include <vector>

#include "src/interfaces/cfd/mooring_line_input.hpp"

namespace openturbine::cfd {

struct FloatingPlatformInput {
    /// Flag to enable use of floating platform in model
    bool enable = false;

    /// Platform point coordinates and orientation (XYZ,quaternion)
    std::array<double, 7> position = {0., 0., 0., 1., 0., 0., 0.};

    /// Platform point translational and rotational velocity
    std::array<double, 6> velocity = {0., 0., 0., 0., 0., 0.};

    /// Platform point translational and rotational acceleration
    std::array<double, 6> acceleration = {0., 0., 0., 0., 0., 0.};

    /// Platform point mass matrix
    std::array<std::array<double, 6>, 6> mass_matrix;

    /// Mooring line array
    std::vector<MooringLineInput> mooring_lines;
};

}