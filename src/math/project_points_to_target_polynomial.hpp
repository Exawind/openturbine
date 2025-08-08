#pragma once

#include <array>
#include <vector>

#include "elements/beams/interpolation.hpp"
#include "least_squares_fit.hpp"

namespace openturbine::math {

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
    // Calculate matrix of num_inputs points-based LSFE shape function values at ouput
    // locations
    const auto shape_functions = ComputeShapeFunctionValues(
        GenerateGLLPoints(num_outputs - 1), GenerateGLLPoints(num_inputs - 1)
    );

    // Project input_points to output locations using LSFE shape functions
    auto output_points = std::vector<std::array<double, 3>>(num_outputs);
    for (auto output = 0U; output < num_outputs; ++output) {
        for (auto input = 0U; input < num_inputs; ++input) {
            for (auto dim = 0U; dim < 3U; ++dim) {
                output_points[output][dim] +=
                    shape_functions[input][output] * input_points[input][dim];
            }
        }
    }
    return output_points;
}

}  // namespace openturbine
