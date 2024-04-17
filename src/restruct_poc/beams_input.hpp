#pragma once

#include <array>

#include "beams.hpp"

#include "src/restruct_poc/types.hpp"

namespace openturbine {

using Array_6x6 = std::array<std::array<double, 6>, 6>;
using Array_2 = std::array<double, 2>;
using Array_3 = std::array<double, kVectorComponents>;
using Array_6 = std::array<double, kLieAlgebraComponents>;
using Array_7 = std::array<double, kLieGroupComponents>;

// Beam node initialization data.
struct BeamNode {
    double s;   // Position of node in element on range [0, 1]
    Array_7 x;  // Node initial positions and rotations

    BeamNode(Array_7 x_) : s(0.), x(x_) {}
    BeamNode(double s_, Array_7 x_) : s(s_), x(x_) {}
    BeamNode(double s_, Vector p, Quaternion q)
        : s(s_), x{p.GetX(),          p.GetY(),          p.GetZ(),         q.GetScalarComponent(),
                   q.GetXComponent(), q.GetYComponent(), q.GetZComponent()} {}
};

// Beam section initialization data
struct BeamSection {
    double s;          // Position of section in element on range [0, 1]
    Array_6x6 M_star;  // Mass matrix in material frame
    Array_6x6 C_star;  // Stiffness matrix in material frame

    BeamSection(double s_, Array_6x6 M_star_, Array_6x6 C_star_)
        : s(s_), M_star(M_star_), C_star(C_star_) {}
};

using BeamQuadrature = std::vector<Array_2>;

struct BeamElement {
    std::vector<BeamNode> nodes;        // Element node positions/rotations in material frame
    std::vector<BeamSection> sections;  // Element mass/stiffness in material frame
    BeamQuadrature quadrature;          // Element quadrature points and weights

    BeamElement(
        std::vector<BeamNode> nodes_, std::vector<BeamSection> sections_, BeamQuadrature quadrature_
    )
        : nodes(nodes_), sections(sections_), quadrature(quadrature_) {
        // // If node positions already set, return
        // if (nodes.back().s_ != 0.)
        //     return;

        // // Calculate distances between nodes in element
        // std::vector<double> node_distances({0.});
        // for (size_t i = 1; i < this->nodes.size(); i++) {
        //     node_distances.push_back(sqrt(
        //         pow(this->nodes[i].x_[0] - this->nodes[i - 1].x_[0], 2) +
        //         pow(this->nodes[i].x_[1] - this->nodes[i - 1].x_[1], 2) +
        //         pow(this->nodes[i].x_[2] - this->nodes[i - 1].x_[2], 2)
        //     ));
        // }

        // // Calculate total element length
        // double length = std::reduce(node_distances.begin(), node_distances.end());

        // // Calculate cumulate distance of nodes
        // std::vector<double> node_cumulative_distances(node_distances.size());
        // std::partial_sum(
        //     node_distances.begin(), node_distances.end(), node_cumulative_distances.begin()
        // );

        // for (size_t i = 0; i < this->nodes.size(); i++) {
        //     this->nodes[i].s_ = node_cumulative_distances[i] / length;
        // }
    }
};

struct BeamsInput {
    std::vector<BeamElement> elements;
    Array_3 gravity;

    BeamsInput(std::vector<BeamElement> elements_, Array_3 gravity_)
        : elements(elements_), gravity(gravity_) {}

    // Returns the number of elements
    size_t NumElements() const { return elements.size(); };

    // Returns the total number of nodes for all elements
    size_t NumNodes() const {
        size_t num_nodes = 0;
        for (const auto& input : this->elements) {
            num_nodes += input.nodes.size();
        }
        return num_nodes;
    }

    // Returns the total number of quadrature points for all elements
    size_t NumQuadraturePoints() const {
        size_t num_qps = 0;
        for (const auto& input : this->elements) {
            num_qps += input.quadrature.size();
        }
        return num_qps;
    }

    // Returns the maximum number of nodes in any element
    size_t MaxElemNodes() const {
        size_t max_elem_nodes = 0;
        for (const auto& input : this->elements) {
            max_elem_nodes = std::max(max_elem_nodes, input.nodes.size());
        }
        return max_elem_nodes;
    }

    // Returns the maximum number of quadrature points in any element
    size_t MaxElemQuadraturePoints() const {
        size_t max_elem_qps = 0;
        for (const auto& input : this->elements) {
            max_elem_qps = std::max(max_elem_qps, input.quadrature.size());
        }
        return max_elem_qps;
    }
};

Beams CreateBeams(const BeamsInput& beams_input);

}  // namespace openturbine
