#pragma once

#include <array>

#include "types.hpp"

namespace openturbine {

struct BeamNode {
    double s;                 // Position of node in element on range [0, 1]
    std::array<double, 7> x;  // Node initial positions and rotations

    BeamNode(std::array<double, 7> x_) : s(0.), x(std::move(x_)) {}
    BeamNode(double s_, std::array<double, 7> x_) : s(s_), x(x_) {}
    BeamNode(double s_, Vector p, Quaternion q)
        : s(s_), x{p.GetX(),          p.GetY(),          p.GetZ(),         q.GetScalarComponent(),
                   q.GetXComponent(), q.GetYComponent(), q.GetZComponent()} {}
};

}  // namespace openturbine
