#pragma once

#include "src/math/quaternion_operations.hpp"
#include "src/types.hpp"

namespace openturbine {

struct Node {
    size_t ID;   //< Node identifier
    Array_7 x;   //< Node positions and orientations
    Array_7 u;   //< Node displacement
    Array_6 v;   //< Node velocity
    Array_6 vd;  //< Node acceleration
    double s;    //< Position of node in element on range [0, 1]

    /// @brief Construct a node with an ID
    Node(size_t id)
        : ID(id),
          x(Array_7{0., 0., 0., 1., 0., 0., 0.}),
          u(Array_7{0., 0., 0., 1., 0., 0., 0.}),
          v(Array_6{0., 0., 0., 0., 0., 0.}),
          vd(Array_6{0., 0., 0., 0., 0., 0.}),
          s(0.) {}

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
            sin(angle / 2.) * axis[2]
        };
        Rotate(q);
    }
};

class NodeBuilder {
public:
    NodeBuilder(Node& n) : node(n) {}

    NodeBuilder& SetPosition(
        const double x, const double y, const double z, const double w, const double i,
        const double j, const double k
    ) {
        this->node.x[0] = x;
        this->node.x[1] = y;
        this->node.x[2] = z;
        this->node.x[3] = w;
        this->node.x[4] = i;
        this->node.x[5] = j;
        this->node.x[6] = k;
        return *this;
    }

    NodeBuilder& SetDisplacement(
        const double x, const double y, const double z, const double w, const double i,
        const double j, const double k
    ) {
        this->node.u[0] = x;
        this->node.u[1] = y;
        this->node.u[2] = z;
        this->node.u[3] = w;
        this->node.u[4] = i;
        this->node.u[5] = j;
        this->node.u[6] = k;
        return *this;
    }

    NodeBuilder& SetVelocity(
        const double x, const double y, const double z, const double rx, const double ry,
        const double rz
    ) {
        this->node.v[0] = x;
        this->node.v[1] = y;
        this->node.v[2] = z;
        this->node.v[3] = rx;
        this->node.v[4] = ry;
        this->node.v[5] = rz;
        return *this;
    }

    NodeBuilder& SetAcceleration(
        const double x, const double y, const double z, const double rx, const double ry,
        const double rz
    ) {
        this->node.vd[0] = x;
        this->node.vd[1] = y;
        this->node.vd[2] = z;
        this->node.vd[3] = rx;
        this->node.vd[4] = ry;
        this->node.vd[5] = rz;
        return *this;
    }

    NodeBuilder& SetElemLocation(double loc) {
        this->node.s = loc;
        return *this;
    }

    size_t Build() { return this->node.ID; }

private:
    Node& node;
};

}  // namespace openturbine
