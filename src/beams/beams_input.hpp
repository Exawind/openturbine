#pragma once

#include <algorithm>
#include <array>
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
        size_t num_nodes{0};
        for (const auto& e : this->elements) {
            num_nodes += e.nodes.size();
        }
        return num_nodes;
    }

    [[nodiscard]] size_t NumQuadraturePoints() const {
        size_t num_qps{0};
        for (const auto& e : this->elements) {
            num_qps += e.quadrature.size();
        }
        return num_qps;
    }

    [[nodiscard]] size_t MaxElemNodes() const {
        size_t max_elem_nodes{0};
        for (const auto& e : this->elements) {
            max_elem_nodes = std::max(max_elem_nodes, e.nodes.size());
        }
        return max_elem_nodes;
    }

    [[nodiscard]] size_t MaxElemQuadraturePoints() const {
        size_t max_elem_qps{0};
        for (const auto& e : this->elements) {
            max_elem_qps = std::max(max_elem_qps, e.quadrature.size());
        }
        return max_elem_qps;
    }
};

}  // namespace openturbine
