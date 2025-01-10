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
    explicit Node(size_t id)
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
        : ID(id), x(position), u(displacement), v(velocity), vd(acceleration), s(0.) {}

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
    explicit NodeBuilder(Node& n) : node(n) {}
    ~NodeBuilder() = default;
    NodeBuilder(const NodeBuilder&) = delete;
    NodeBuilder(NodeBuilder&&) = delete;
    NodeBuilder& operator=(const NodeBuilder&) = delete;
    NodeBuilder& operator=(NodeBuilder&&) = delete;

    //--------------------------------------------------------------------------
    // Set position
    //--------------------------------------------------------------------------

    NodeBuilder& SetPosition(const Array_7& p) {
        this->node.x = p;
        return *this;
    }

    NodeBuilder& SetPosition(double x, double y, double z, double w, double i, double j, double k) {
        this->node.x = {x, y, z, w, i, j, k};
        return *this;
    }

    NodeBuilder& SetPosition(const Array_3& p) {
        this->node.x[0] = p[0];
        this->node.x[1] = p[1];
        this->node.x[2] = p[2];
        return *this;
    }

    NodeBuilder& SetPosition(double x, double y, double z) {
        this->node.x[0] = x;
        this->node.x[1] = y;
        this->node.x[2] = z;
        return *this;
    }

    //--------------------------------------------------------------------------
    // Set orientation
    //--------------------------------------------------------------------------

    /// @brief Sets the node orientation from quaternion
    /// @param p quaternion (w,i,j,k)
    NodeBuilder& SetOrientation(const Array_4& p) {
        this->node.x[3] = p[0];
        this->node.x[4] = p[1];
        this->node.x[5] = p[2];
        this->node.x[6] = p[3];
        return *this;
    }

    /// @brief Sets the node orientation from quaternion components
    /// @param p quaternion (w,i,j,k)
    NodeBuilder& SetOrientation(double w, double i, double j, double k) {
        this->node.x[3] = w;
        this->node.x[4] = i;
        this->node.x[5] = j;
        this->node.x[6] = k;
        return *this;
    }

    //--------------------------------------------------------------------------
    // Set displacement
    //--------------------------------------------------------------------------

    NodeBuilder& SetDisplacement(const Array_7& p) {
        this->node.u = p;
        return *this;
    }

    NodeBuilder& SetDisplacement(
        double x, double y, double z, double w, double i, double j, double k
    ) {
        this->node.u = {x, y, z, w, i, j, k};
        return *this;
    }

    NodeBuilder& SetDisplacement(const Array_3& p) {
        this->node.u[0] = p[0];
        this->node.u[1] = p[1];
        this->node.u[2] = p[2];
        return *this;
    }

    NodeBuilder& SetDisplacement(double x, double y, double z) {
        this->node.u[0] = x;
        this->node.u[1] = y;
        this->node.u[2] = z;
        return *this;
    }

    //--------------------------------------------------------------------------
    // Set velocity
    //--------------------------------------------------------------------------

    NodeBuilder& SetVelocity(double x, double y, double z, double rx, double ry, double rz) {
        this->node.v = {x, y, z, rx, ry, rz};
        return *this;
    }

    NodeBuilder& SetVelocity(const Array_6& v) {
        this->node.v = v;
        return *this;
    }

    NodeBuilder& SetVelocity(double x, double y, double z) {
        this->node.v[0] = x;
        this->node.v[1] = y;
        this->node.v[2] = z;
        return *this;
    }

    NodeBuilder& SetVelocity(const Array_3& v) {
        this->node.v[0] = v[0];
        this->node.v[1] = v[1];
        this->node.v[2] = v[2];
        return *this;
    }

    //--------------------------------------------------------------------------
    // Set acceleration
    //--------------------------------------------------------------------------

    NodeBuilder& SetAcceleration(double x, double y, double z, double rx, double ry, double rz) {
        this->node.vd = {x, y, z, rx, ry, rz};
        return *this;
    }

    NodeBuilder& SetAcceleration(const Array_6& v) {
        this->node.vd = v;
        return *this;
    }

    NodeBuilder& SetAcceleration(double x, double y, double z) {
        this->node.vd[0] = x;
        this->node.vd[1] = y;
        this->node.vd[2] = z;
        return *this;
    }

    NodeBuilder& SetAcceleration(const Array_3& v) {
        this->node.vd[0] = v[0];
        this->node.vd[1] = v[1];
        this->node.vd[2] = v[2];
        return *this;
    }

    //--------------------------------------------------------------------------
    // Element location
    //--------------------------------------------------------------------------

    NodeBuilder& SetElemLocation(double loc) {
        this->node.s = loc;
        return *this;
    }

    //--------------------------------------------------------------------------
    // Build
    //--------------------------------------------------------------------------

    /// Build finalizes construction of node and returns the node's ID
    [[nodiscard]] size_t Build() const { return this->node.ID; }

private:
    Node& node;
};

}  // namespace openturbine
