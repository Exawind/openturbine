#pragma once

#include "src/restruct_poc/math/quaternion_operations.hpp"
#include "src/restruct_poc/types.hpp"

namespace openturbine {

struct Node {
    size_t ID;   //< Node identifier
    Array_7 x;   //< Node positions and orientations
    Array_7 u;   //< Node displacement
    Array_6 v;   //< Node velocity
    Array_6 vd;  //< Node acceleration

    /// @brief Construct a node with an ID, position, displacement, velocity, and acceleration
    /// vectors
    Node(
        size_t id, Array_7 position, Array_7 displacement = Array_7{0., 0., 0., 1., 0., 0., 0.},
        Array_6 velocity = Array_6{0., 0., 0., 0., 0., 0.},
        Array_6 acceleration = Array_6{0., 0., 0., 0., 0., 0.}
    )
        : ID(id), x(position), u(displacement), v(velocity), vd(acceleration) {}

    /// Translate node by a displacement vector
    void Translate(const Array_3& displacement) {
        x[0] += displacement[0];
        x[1] += displacement[1];
        x[2] += displacement[2];
    }

    /// Rotate node by a quaternion
    void Rotate(const Array_4& q) {
        // Rotate position
        auto x_rot = RotateVectorByQuaternion(q, {x[0], x[1], x[2]});
        x[0] = x_rot[0];
        x[1] = x_rot[1];
        x[2] = x_rot[2];

        // Rotate orientation
        auto q_rot = QuaternionCompose(q, {x[3], x[4], x[5], x[6]});
        x[3] = q_rot[0];
        x[4] = q_rot[1];
        x[5] = q_rot[2];
        x[6] = q_rot[3];
    }

    /// Rotate node by a rotation axis and angle
    void Rotate(const Array_3& axis, double angle) {
        auto q = Array_4{
            cos(angle / 2.), sin(angle / 2.) * axis[0], sin(angle / 2.) * axis[1],
            sin(angle / 2.) * axis[2]};
        Rotate(q);
    }
};

}  // namespace openturbine