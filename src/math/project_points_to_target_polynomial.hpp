#pragma once

#include <array>
#include <vector>

#include "elements/beams/interpolation.hpp"
#include "least_squares_fit.hpp"

namespace openturbine {

/**
 * @brief Projects 3D points from a given (lower) polynomial representation to a target (higher)
 * polynomial representation
 *
 * This function maps a set of 3D points defined at nodes of a polynomial of order source_order
 * to corresponding points at nodes of a polynomial of order target_order (typically higher
 * than the source order) using Least-Squares Finite Element (LSFE) shape functions.
 *
 * Primary use case: The primary application of this function is to increase the number of
 * points from a lower-order geometric representation to the higher-order representation
 * required for spectral finite element analysis. This enables the use of high-order methods
 * while allowing geometry to be defined with fewer points initially.
 *
 * @param num_inputs Number of points in the source polynomial representation
 * @param num_outputs Number of points in the target polynomial representation
 * @param input_points 3D coordinates of points in the source representation
 * @return std::vector<std::array<double, 3>>
 *         - Coordinates of the projected 3D points at the target polynomial nodes
 */
inline std::vector<std::array<double, 3>> ProjectPointsToTargetPolynomial(
    size_t num_inputs, size_t num_outputs, const std::vector<std::array<double, 3>>& input_points
) {
    // Step 1: Calculate locations of GLL points (the order will be num_outputs - 1)
    auto output_gll_pts = GenerateGLLPoints(num_outputs - 1);

    // Step 2: Calculate matrix of num_inputs points-based LSFE shape function values at ouput
    // locations
    [[maybe_unused]] const auto [shape_functions, shape_derivatives, gll_pts] =
        ShapeFunctionMatrices(num_outputs, num_inputs, output_gll_pts);

    // Step 3: Project input_points to output locations using LSFE shape functions
    auto output_points = std::vector<std::array<double, 3>>(num_outputs);
    for (size_t i_output = 0; i_output < num_outputs; ++i_output) {
        for (size_t i_input = 0; i_input < num_inputs; ++i_input) {
            for (size_t dim = 0; dim < 3; ++dim) {
                output_points[i_output][dim] +=
                    shape_functions[i_input][i_output] * input_points[i_input][dim];
            }
        }
    }
    return output_points;
}

}  // namespace openturbine
