#pragma once

#include <array>

namespace openturbine {

struct MassNode {
    std::array<double, 7> initial_dofs;  // Node initial positions and rotations

    MassNode(std::array<double, 7> x) : initial_dofs(std::move(x)) {}
    MassNode(std::array<double, 3> p, std::array<double, 4> q)
        : initial_dofs{p[0], p[1], p[2], q[0], q[1], q[2], q[3]} {}
};

}  // namespace openturbine
