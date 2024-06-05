#pragma once

#include <array>
#include <vector>

#include "mass_node.hpp"

#include "src/restruct_poc/types.hpp"

namespace openturbine {

struct MassElement {
    MassNode node;     // Element node positions/rotations in material frame
    Array_6x6 M_star;  // Mass matrix in material frame

    MassElement(MassNode n, Array_6x6 mass) : node(std::move(n)), M_star(std::move(mass)) {}
    MassElement(MassNode n, double mass, std::array<double, 3> J)
        : node(std::move(n)),
          M_star(Array_6x6{{
              {mass, 0., 0., 0., 0., 0.},
              {0., mass, 0., 0., 0., 0.},
              {0., 0., mass, 0., 0., 0.},
              {0., 0., 0., J[0], 0., 0.},
              {0., 0., 0., 0., J[1], 0.},
              {0., 0., 0., 0., 0., J[2]},
          }}) {}
};

}  // namespace openturbine
