#pragma once

#include <array>

namespace openturbine {

struct BeamNode {
    double position;                     // Position of node in element on range [0, 1]
    std::array<double, 7> initial_dofs;  // Node initial positions and rotations

    BeamNode(std::array<double, 7> x) : position(0.), initial_dofs(std::move(x)) {}
    BeamNode(double s, std::array<double, 7> x) : position(s), initial_dofs(x) {}
    BeamNode(double s, std::array<double, 3> p, std::array<double, 4> q)
        : position(s), initial_dofs{p[0], p[1], p[2], q[0], q[1], q[2], q[3]} {}
};

}  // namespace openturbine
