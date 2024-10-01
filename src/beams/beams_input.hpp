#pragma once

#include <algorithm>
#include <array>
#include <numeric>
#include <vector>

#include "beam_element.hpp"

namespace openturbine {

struct BeamsInput {
    std::vector<BeamElement> elements;
    std::array<double, 3> gravity;

    BeamsInput(std::vector<BeamElement> elems, std::array<double, 3> g)
        : elements(std::move(elems)), gravity(g) {}

    [[nodiscard]] size_t NumElements() const { return elements.size(); };

    [[nodiscard]] size_t NumNodes() const {
        return std::transform_reduce(
            elements.begin(), elements.end(), size_t{0U}, std::plus{},
            [](const BeamElement& e) {
                return e.nodes.size();
            }
        );
    }

    [[nodiscard]] size_t NumQuadraturePoints() const {
        return std::transform_reduce(
            elements.begin(), elements.end(), size_t{0U}, std::plus{},
            [](const BeamElement& e) {
                return e.quadrature.size();
            }
        );
    }

    [[nodiscard]] size_t MaxElemNodes() const {
        return std::transform_reduce(
            elements.begin(), elements.end(), size_t{0U},
            [](size_t a, size_t b) {
                return std::max(a, b);
            },
            [](const BeamElement& e) {
                return e.nodes.size();
            }
        );
    }

    [[nodiscard]] size_t MaxElemQuadraturePoints() const {
        return std::transform_reduce(
            elements.begin(), elements.end(), size_t{0U},
            [](size_t a, size_t b) {
                return std::max(a, b);
            },
            [](const BeamElement& e) {
                return e.quadrature.size();
            }
        );
    }
};

}  // namespace openturbine
