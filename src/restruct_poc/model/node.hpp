#pragma once

#include "src/restruct_poc/math/quaternion_operations.hpp"
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

    /// Translate a node by a displacement vector
    void Translate(const Array_3& displacement) {
        x[0] += displacement[0];
        x[1] += displacement[1];
        x[2] += displacement[2];
    }

    /// Rotate a node by a rotation axis and angle
    void Rotate(const Array_3& axis, double angle) {
        auto q = Array_4{
            cos(angle / 2.), sin(angle / 2.) * axis[0], sin(angle / 2.) * axis[1],
            sin(angle / 2.) * axis[2]};
        auto R = QuaternionToRotationMatrix(q);
        auto x_rot = std::array<double, 3>{};
        for (int i = 0; i < 3; ++i) {
            x_rot[i] = R[i][0] * x[0] + R[i][1] * x[1] + R[i][2] * x[2];
        }
        x[0] = x_rot[0];
        x[1] = x_rot[1];
        x[2] = x_rot[2];
    }
};

}  // namespace openturbine