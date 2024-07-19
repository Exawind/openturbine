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

    // Rotate velocity by a quaternion
    void RotateVelocity(const Array_4& q) {
        auto v_rot = RotateVectorByQuaternion(q, {v[3], v[4], v[5]});
        v[3] = v_rot[0];
        v[4] = v_rot[1];
        v[5] = v_rot[2];
    }

    // Rotate velocity by a rotation axis and angle
    void RotateVelocity(const Array_3& axis, double angle) {
        auto q = Array_4{
            cos(angle / 2.), sin(angle / 2.) * axis[0], sin(angle / 2.) * axis[1],
            sin(angle / 2.) * axis[2]};
        RotateVelocity(q);
    }
};

}  // namespace openturbine