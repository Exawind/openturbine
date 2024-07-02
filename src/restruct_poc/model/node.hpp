#pragma once

#include "src/restruct_poc/types.hpp"

namespace openturbine {

struct Node {
    int ID;      // Node identifier
    Array_7 x;   // Node positions and orientations
    Array_7 u;   // Node displacement
    Array_6 v;   // Node velocity
    Array_6 vd;  // Node acceleration

    Node(
        int id, Array_7 position, Array_7 displacement = Array_7{0., 0., 0., 1., 0., 0., 0.},
        Array_6 velocity = Array_6{0., 0., 0., 0., 0., 0.},
        Array_6 acceleration = Array_6{0., 0., 0., 0., 0., 0.}
    )
        : ID(id), x(position), u(displacement), v(velocity), vd(acceleration) {}
};

}  // namespace openturbine