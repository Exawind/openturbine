#pragma once

#include "types.hpp"

namespace openturbine::interfaces::components {

struct ReferenceAxis {
    /// @brief Coordinate locations [0, 1]
    std::vector<double> coordinate_grid;

    /// @brief X,Y,Z coordinates of reference axis points
    std::vector<std::array<double, 3>> coordinates;

    /// @brief Twist locations [0, 1]
    std::vector<double> twist_grid;

    /// @brief Structural twist
    std::vector<double> twist;
};

struct Section {
    /// @brief Section locations [0, 1]
    double location;

    /// @brief Mass matrix
    Array_6x6 mass_matrix;

    /// @brief Stiffness matrix
    Array_6x6 stiffness_matrix;

    Section(double loc, Array_6x6 m, Array_6x6 k)
        : location(loc), mass_matrix(m), stiffness_matrix(k) {}
};

struct Root {
    /// @brief Flag to use root motion as an input to the model
    bool prescribe_root_motion{false};

    /// @brief Inital position/orientation
    std::array<double, 7> position{0., 0., 0., 1., 0., 0., 0.};

    /// @brief Initial translation/rotational velocity
    std::array<double, 6> velocity{0., 0., 0., 0., 0., 0.};

    /// @brief Initial translation/rotational acceleration
    std::array<double, 6> acceleration{0., 0., 0., 0., 0., 0.};
};

struct BladeInput {
    /// @brief Spectral element order (num nodes - 1)
    size_t element_order{10};

    /// @brief Trapezoidal quadrature point refinement (0 = none)
    size_t section_refinement{0};

    /// @brief Structural reference axis data
    ReferenceAxis ref_axis;

    /// @brief Blade root
    Root root;

    /// @brief Section properties
    std::vector<Section> sections;
};

}  // namespace openturbine::interfaces::components
