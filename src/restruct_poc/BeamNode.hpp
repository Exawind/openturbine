#pragma once

#include <array>

namespace openturbine {

struct BeamNode {
    double s;                 // Position of node in element on range [0, 1]
    std::array<double, 7> x;  // Node initial positions and rotations

    BeamNode(std::array<double, 7> x_) : s(0.), x(std::move(x_)) {}
    BeamNode(double s_, std::array<double, 7> x_) : s(s_), x(x_) {}
    BeamNode(double s_, std::array<double, 3> p, std::array<double, 4> q)
        : s(s_), x{p[0], p[1], p[2], q[0], q[1], q[2], q[3]} {}
};

}  // namespace openturbine
