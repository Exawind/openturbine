#pragma once

#include <vector>
#include <array>
#include <algorithm>

#include "BeamElement.hpp"

namespace openturbine {

struct BeamsInput {
    std::vector<BeamElement> elements;
    std::array<double, 3> gravity;

    BeamsInput(std::vector<BeamElement> elements_, std::array<double, 3> gravity_)
        : elements(std::move(elements_)), gravity(std::move(gravity_)) {}

    size_t NumElements() const { return elements.size(); };

    size_t NumNodes() const {
        size_t num_nodes = 0;
        for (const auto& input : this->elements) {
            num_nodes += input.nodes.size();
        }
        return num_nodes;
    }   

    size_t NumQuadraturePoints() const {
        size_t num_qps = 0;
        for (const auto& input : this->elements) {
            num_qps += input.quadrature.size();
        }
        return num_qps;
    }

    size_t MaxElemNodes() const {
        size_t max_elem_nodes = 0;
        for (const auto& input : this->elements) {
            max_elem_nodes = std::max(max_elem_nodes, input.nodes.size());
        }
        return max_elem_nodes;
    }

    size_t MaxElemQuadraturePoints() const {
        size_t max_elem_qps = 0;
        for (const auto& input : this->elements) {
            max_elem_qps = std::max(max_elem_qps, input.quadrature.size());
        }
        return max_elem_qps;
    }
};

}
