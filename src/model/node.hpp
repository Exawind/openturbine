#pragma once

#include "dof_management/freedom_signature.hpp"
#include "math/quaternion_operations.hpp"
#include "types.hpp"

namespace openturbine {

/*
 * @brief Represents a node in the finite element model
 *
 * @details A node is a point in 3D space that can have position, orientation, displacement,
 * velocity, and acceleration. Nodes are the fundamental building blocks that connect elements
 * in the structural model. Each node has:
 * - Unique identifier (id) -> used to reference the node in elements and constraints
 * - Initial position and orientation (x0) -> 7 x 1 vector
 * - Displacement from initial position (u) -> 7 x 1 vector
 * - Velocity (v) -> 6 x 1 vector
 * - Acceleration (vd) -> 6 x 1 vector
 * - Parametric position within an element (s) -> scalar
 *
 * The position and displacement vectors contain 7 components: [x, y, z, qw, qx, qy, qz]
 * where the first 3 are translational and the last 4 represent orientation as a quaternion.
 *
 * The velocity and acceleration vectors contain 6 components: [vx, vy, vz, wx, wy, wz]
 * where the first 3 are translational and the last 3 are rotational components.
 */
struct Node {
    size_t id;   //< Node identifier
    Array_7 x0;  //< Initial node positions and orientations
    Array_7 u;   //< Node displacement
    Array_6 v;   //< Node velocity
    Array_6 vd;  //< Node acceleration
    double s;    //< Position of node in element on range [0, 1]

    /// @brief Construct a node with an ID
    explicit Node(size_t node_id)
        : id(node_id),
          x0(Array_7{0., 0., 0., 1., 0., 0., 0.}),
          u(Array_7{0., 0., 0., 1., 0., 0., 0.}),
          v(Array_6{0., 0., 0., 0., 0., 0.}),
          vd(Array_6{0., 0., 0., 0., 0., 0.}),
          s(0.) {}

    /// @brief Construct a node with an ID, position, displacement, velocity, and acceleration
    /// vectors
    Node(
        size_t node_id, Array_7 position, Array_7 displacement = Array_7{0., 0., 0., 1., 0., 0., 0.},
        Array_6 velocity = Array_6{0., 0., 0., 0., 0., 0.},
        Array_6 acceleration = Array_6{0., 0., 0., 0., 0., 0.}
    )
        : id(node_id), x0(position), u(displacement), v(velocity), vd(acceleration), s(0.) {}

    //--------------------------------------------------------------------------
    // Compute displaced/current position
    //--------------------------------------------------------------------------

    /*
     * @brief Get the displaced position (initial position + displacement)
     * @return Array_7 containing displaced position and orientation
     */
    [[nodiscard]] Array_7 DisplacedPosition() const {
        Array_7 displaced_position{0., 0., 0., 1., 0., 0., 0.};

        // Add translational components (x, y, z)
        displaced_position[0] = this->x0[0] + this->u[0];
        displaced_position[1] = this->x0[1] + this->u[1];
        displaced_position[2] = this->x0[2] + this->u[2];

        // Compose quaternions for orientation (w, i, j, k)
        auto q_displaced = QuaternionCompose(
            {this->x0[3], this->x0[4], this->x0[5], this->x0[6]},  // initial orientation
            {this->u[3], this->u[4], this->u[5], this->u[6]}       // displacement orientation
        );
        displaced_position[3] = q_displaced[0];
        displaced_position[4] = q_displaced[1];
        displaced_position[5] = q_displaced[2];
        displaced_position[6] = q_displaced[3];

        return displaced_position;
    }

    //--------------------------------------------------------------------------
    // Modify node position (x)
    //--------------------------------------------------------------------------

    /// Translate node by a displacement vector
    void Translate(const Array_3& displacement) {
        this->x0[0] += displacement[0];
        this->x0[1] += displacement[1];
        this->x0[2] += displacement[2];
    }

    /// Rotate node by a quaternion about the given point
    void RotateAboutPoint(const Array_4& q, const Array_3& point) {
        // Rotate position i.e. x(0:2)
        auto x_new = RotateVectorByQuaternion(
            q, {this->x0[0] - point[0], this->x0[1] - point[1], this->x0[2] - point[2]}
        );
        this->x0[0] = x_new[0] + point[0];
        this->x0[1] = x_new[1] + point[1];
        this->x0[2] = x_new[2] + point[2];

        // Rotate orientation i.e. x(3:6)
        auto q_new = QuaternionCompose(q, {this->x0[3], this->x0[4], this->x0[5], this->x0[6]});
        this->x0[3] = q_new[0];
        this->x0[4] = q_new[1];
        this->x0[5] = q_new[2];
        this->x0[6] = q_new[3];
    }

    /// Rotate node by a rotation vector about the given point
    void RotateAboutPoint(const Array_3& rv, const Array_3& point) {
        const auto q = RotationVectorToQuaternion(rv);
        this->RotateAboutPoint(q, point);
    }

    //--------------------------------------------------------------------------
    // Modify node displacement (u)
    //--------------------------------------------------------------------------

    /// Add translational displacement to node displacement vector
    void TranslateDisplacement(const Array_3& displacement) {
        this->u[0] += displacement[0];
        this->u[1] += displacement[1];
        this->u[2] += displacement[2];
    }

    /// Rotate node displacement by a quaternion about the given point
    void RotateDisplacementAboutPoint(const Array_4& q, const Array_3& point) {
        // Rotate displacement position i.e. u(0:2)
        auto u_new = RotateVectorByQuaternion(
            q, {this->u[0] - point[0], this->u[1] - point[1], this->u[2] - point[2]}
        );
        this->u[0] = u_new[0] + point[0];
        this->u[1] = u_new[1] + point[1];
        this->u[2] = u_new[2] + point[2];

        // Rotate displacement orientation i.e. u(3:6)
        auto q_new = QuaternionCompose(q, {this->u[3], this->u[4], this->u[5], this->u[6]});
        this->u[3] = q_new[0];
        this->u[4] = q_new[1];
        this->u[5] = q_new[2];
        this->u[6] = q_new[3];
    }

    /// Rotate node displacement by a rotation vector about the given point
    void RotateDisplacementAboutPoint(const Array_3& rv, const Array_3& point) {
        const auto q = RotationVectorToQuaternion(rv);
        this->RotateDisplacementAboutPoint(q, point);
    }

    //--------------------------------------------------------------------------
    // Modify node velocity (v)
    //--------------------------------------------------------------------------

    /// Set node velocity based on rigid body motion about a reference point
    void SetVelocityAboutPoint(const Array_6& velocity, const Array_3& point) {
        // Get distance from reference point to displaced node position
        const auto displaced_position = this->DisplacedPosition();
        const Array_3 r{
            displaced_position[0] - point[0], displaced_position[1] - point[1],
            displaced_position[2] - point[2]
        };

        // Calculate velocity contribution from angular velocity
        const Array_3 omega{velocity[3], velocity[4], velocity[5]};
        const auto omega_cross_r = CrossProduct(omega, r);

        // Set node translational velocity
        this->v[0] = velocity[0] + omega_cross_r[0];
        this->v[1] = velocity[1] + omega_cross_r[1];
        this->v[2] = velocity[2] + omega_cross_r[2];

        // Set node angular velocity
        this->v[3] = velocity[3];
        this->v[4] = velocity[4];
        this->v[5] = velocity[5];
    }

    //--------------------------------------------------------------------------
    // Modify node acceleration (vd)
    //--------------------------------------------------------------------------

    /// Set node acceleration based on rigid body motion about a reference point
    void SetAccelerationAboutPoint(
        const Array_6& acceleration, const Array_3& omega, const Array_3& point
    ) {
        // Get distance from reference point to displaced node position
        const auto displaced_position = this->DisplacedPosition();
        const Array_3 r{
            displaced_position[0] - point[0], displaced_position[1] - point[1],
            displaced_position[2] - point[2]
        };

        // Calculate translational acceleration contribution from angular velocity
        const Array_3 alpha{acceleration[3], acceleration[4], acceleration[5]};
        const auto alpha_cross_r = CrossProduct(alpha, r);
        const auto omega_cross_omega_cross_r = CrossProduct(omega, CrossProduct(omega, r));

        // Set node translational acceleration
        this->vd[0] = acceleration[0] + alpha_cross_r[0] + omega_cross_omega_cross_r[0];
        this->vd[1] = acceleration[1] + alpha_cross_r[1] + omega_cross_omega_cross_r[1];
        this->vd[2] = acceleration[2] + alpha_cross_r[2] + omega_cross_omega_cross_r[2];

        // Set node angular acceleration
        this->vd[3] = alpha[0];
        this->vd[4] = alpha[1];
        this->vd[5] = alpha[2];
    }
};

/**
 * @brief Builder class for constructing and configuring Node objects
 *
 * @details NodeBuilder implements the builder pattern to provide a fluent interface
 * for setting node properties. It allows for method chaining to configure multiple
 * node properties in a single expression, making node creation more readable and
 * maintainable.
 *
 * The builder operates on a reference to an existing Node object and provides
 * methods to set:
 * - Position (initial position and orientation)
 * - Displacement (from initial position)
 * - Velocity (translational and rotational)
 * - Acceleration (translational and rotational)
 * - Element location (parametric position within an element)
 *
 * @note The builder holds a reference to the node being constructed and is not
 * copyable or movable to prevent accidental misuse. All setter methods return
 * a reference to the builder to enable method chaining.
 */
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

    /*
     * @brief Sets the node position from a 7 x 1 vector
     * @param p -> 7 x 1 vector (x, y, z, w, i, j, k)
     */
    NodeBuilder& SetPosition(const Array_7& p) {
        this->node.x0 = p;
        return *this;
    }

    /*
     * @brief Sets the node position from position and orientation components
     * @param x position X component
     * @param y position Y component
     * @param z position Z component
     * @param w quaternion w component (scalar part)
     * @param i quaternion i component (x vector part)
     * @param j quaternion j component (y vector part)
     * @param k quaternion k component (z vector part)
     */
    NodeBuilder& SetPosition(double x, double y, double z, double w, double i, double j, double k) {
        return this->SetPosition(Array_7{x, y, z, w, i, j, k});
    }

    /*
     * @brief Sets the node position from translational components
     * @param x X position component
     * @param y Y position component
     * @param z Z position component
     */
    NodeBuilder& SetPosition(const Array_3& p) {
        this->node.x0[0] = p[0];
        this->node.x0[1] = p[1];
        this->node.x0[2] = p[2];
        return *this;
    }

    /*
     * @brief Sets the node position from translational components
     * @param x X position component
     * @param y Y position component
     * @param z Z position component
     */
    NodeBuilder& SetPosition(double x, double y, double z) {
        return this->SetPosition(Array_3{x, y, z});
    }

    //--------------------------------------------------------------------------
    // Set orientation
    //--------------------------------------------------------------------------

    /*
     * @brief Sets the node orientation from quaternion
     * @param p quaternion (w,i,j,k)
     */
    NodeBuilder& SetOrientation(const Array_4& p) {
        this->node.x0[3] = p[0];
        this->node.x0[4] = p[1];
        this->node.x0[5] = p[2];
        this->node.x0[6] = p[3];
        return *this;
    }

    /*
     * @brief Sets the node orientation from quaternion components
     * @param w quaternion w component (scalar part)
     * @param i quaternion i component (x vector part)
     * @param j quaternion j component (y vector part)
     * @param k quaternion k component (z vector part)
     */
    NodeBuilder& SetOrientation(double w, double i, double j, double k) {
        return this->SetOrientation(Array_4{w, i, j, k});
    }

    //--------------------------------------------------------------------------
    // Set displacement
    //--------------------------------------------------------------------------

    /*
     * @brief Sets the node displacement from a 7 x 1 vector
     * @param p -> 7 x 1 vector (x, y, z, w, i, j, k)
     */
    NodeBuilder& SetDisplacement(const Array_7& p) {
        this->node.u = p;
        return *this;
    }

    /*
     * @brief Sets the node displacement from displacement components
     * @param x displacement X component
     * @param y displacement Y component
     * @param z displacement Z component
     * @param w quaternion w component (scalar part)
     * @param i quaternion i component (x vector part)
     * @param j quaternion j component (y vector part)
     * @param k quaternion k component (z vector part)
     */
    NodeBuilder& SetDisplacement(
        double x, double y, double z, double w, double i, double j, double k
    ) {
        return this->SetDisplacement(Array_7{x, y, z, w, i, j, k});
    }

    /*
     * @brief Sets the node displacement from translational components
     * @param x displacement X component
     * @param y displacement Y component
     * @param z displacement Z component
     */
    NodeBuilder& SetDisplacement(const Array_3& p) {
        this->node.u[0] = p[0];
        this->node.u[1] = p[1];
        this->node.u[2] = p[2];
        return *this;
    }

    /*
     * @brief Sets the node displacement from translational components
     * @param x displacement X component
     * @param y displacement Y component
     * @param z displacement Z component
     */
    NodeBuilder& SetDisplacement(double x, double y, double z) {
        return this->SetDisplacement(Array_3{x, y, z});
    }

    //--------------------------------------------------------------------------
    // Set velocity
    //--------------------------------------------------------------------------

    /*
     * @brief Sets the node velocity from 6 vector components
     * @param x x-component of translational velocity
     * @param y y-component of translational velocity
     * @param z z-component of translational velocity
     * @param rx x-component of rotational velocity
     * @param ry y-component of rotational velocity
     * @param rz z-component of rotational velocity
     */
    NodeBuilder& SetVelocity(double x, double y, double z, double rx, double ry, double rz) {
        this->node.v = {x, y, z, rx, ry, rz};
        return *this;
    }

    /*
     * @brief Sets the node velocity from a vector
     * @param v -> 6 x 1 vector (x, y, z, rx, ry, rz)
     */
    NodeBuilder& SetVelocity(const Array_6& v) {
        return this->SetVelocity(v[0], v[1], v[2], v[3], v[4], v[5]);
    }

    /*
     * @brief Sets the node velocity from 3 vector components
     * @param x x-component of translational velocity
     * @param y y-component of translational velocity
     * @param z z-component of translational velocity
     */
    NodeBuilder& SetVelocity(double x, double y, double z) {
        this->node.v[0] = x;
        this->node.v[1] = y;
        this->node.v[2] = z;
        return *this;
    }

    /*
     * @brief Sets the node velocity from a 3 x 1 vector
     * @param v -> 3D vector (x, y, z)
     */
    NodeBuilder& SetVelocity(const Array_3& v) { return this->SetVelocity(v[0], v[1], v[2]); }

    //--------------------------------------------------------------------------
    // Set acceleration
    //--------------------------------------------------------------------------

    /*
     * @brief Sets the node acceleration from 6 vector components
     * @param x x-component of translational acceleration
     * @param y y-component of translational acceleration
     * @param z z-component of translational acceleration
     * @param rx x-component of rotational acceleration
     * @param ry y-component of rotational acceleration
     * @param rz z-component of rotational acceleration
     */
    NodeBuilder& SetAcceleration(double x, double y, double z, double rx, double ry, double rz) {
        this->node.vd = {x, y, z, rx, ry, rz};
        return *this;
    }

    /*
     * @brief Sets the node acceleration from a vector
     * @param v -> 6 x 1 vector (x, y, z, rx, ry, rz)
     */
    NodeBuilder& SetAcceleration(const Array_6& v) {
        return this->SetAcceleration(v[0], v[1], v[2], v[3], v[4], v[5]);
    }

    /*
     * @brief Sets the node acceleration from vector components
     * @param x x-component of acceleration
     * @param y y-component of acceleration
     * @param z z-component of acceleration
     */
    NodeBuilder& SetAcceleration(double x, double y, double z) {
        this->node.vd[0] = x;
        this->node.vd[1] = y;
        this->node.vd[2] = z;
        return *this;
    }

    /*
     * @brief Sets the node acceleration from a vector
     * @param v -> 3 x 1 vector (x, y, z)
     */
    NodeBuilder& SetAcceleration(const Array_3& v) {
        return this->SetAcceleration(v[0], v[1], v[2]);
    }

    //--------------------------------------------------------------------------
    // Element location
    //--------------------------------------------------------------------------

    /*
     * @brief Sets the parametric position of the node within the element
     * @param location -> position within element on range [0, 1]
     */
    NodeBuilder& SetElemLocation(double location) {
        this->node.s = location;
        return *this;
    }

    //--------------------------------------------------------------------------
    // Build
    //--------------------------------------------------------------------------

    /// Build finalizes construction of node and returns the node's ID
    [[nodiscard]] size_t Build() const { return this->node.id; }

private:
    Node& node;
};

}  // namespace openturbine
