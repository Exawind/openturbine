#pragma once

#include <algorithm>
#include <array>
#include <numeric>
#include <vector>

#include "beam_element.hpp"

namespace openturbine {

/**
 * @brief Represents the input data for beam simulations
 *
 * This struct encapsulates all necessary input parameters for beam simulations,
 * including the beam elements and environmental factors such as gravity.
 * It also provides utility methods for accessing and computing various
 * properties of the beam structure.
 */
struct BeamsInput {
    std::vector<BeamElement> elements;  //< Elements in the beam
    std::array<double, 3> gravity;      //< Gravity vector

    BeamsInput(std::vector<BeamElement> elems, std::array<double, 3> g)
        : elements(std::move(elems)), gravity(g) {}

    /// Returns the number of elements in the beam
    [[nodiscard]] size_t NumElements() const { return elements.size(); }

    /// Computes the sum of a value across all elements
    template <typename Accessor>
    [[nodiscard]] size_t ComputeSum(Accessor accessor) const {
        return std::transform_reduce(
            elements.begin(), elements.end(), size_t{0U}, std::plus{},
            [&accessor](const BeamElement& e) {
                return accessor(e);
            }
        );
    }

    /// Computes the maximum of a value across all elements
    template <typename Accessor>
    [[nodiscard]] size_t ComputeMax(Accessor accessor) const {
        return std::transform_reduce(
            elements.begin(), elements.end(), size_t{0U},
            [](auto a, auto b) {
                return std::max(a, b);
            },
            [&accessor](const auto& e) {
                return accessor(e);
            }
        );
    }

    /// Returns the total number of nodes in the beam
    [[nodiscard]] size_t NumNodes() const {
        return ComputeSum([](const BeamElement& e) {
            return e.nodes.size();
        });
    }

    /// Returns the total number of quadrature points in the beam
    [[nodiscard]] size_t NumQuadraturePoints() const {
        return ComputeSum([](const BeamElement& e) {
            return e.quadrature.size();
        });
    }

    /// Returns the maximum number of nodes in any element of the beam
    [[nodiscard]] size_t MaxElemNodes() const {
        return ComputeMax([](const BeamElement& e) {
            return e.nodes.size();
        });
    }

    /// Returns the maximum number of quadrature points in any element of the beam
    [[nodiscard]] size_t MaxElemQuadraturePoints() const {
        return ComputeMax([](const BeamElement& e) {
            return e.quadrature.size();
        });
    }
};

}  // namespace openturbine
