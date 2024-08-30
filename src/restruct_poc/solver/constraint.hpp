#pragma once

#include "src/restruct_poc/math/vector_operations.hpp"
#include "src/restruct_poc/model/node.hpp"

namespace openturbine {

/// @brief Enum class to define the type of the constraint
enum class ConstraintType : std::uint8_t {
    kNone = 0,           // No constraint (default)
    kFixedBC = 1,        // Fixed boundary condition/clamped constraint (all 6 DOFs fixed)
    kPrescribedBC = 2,   // Prescribed boundary condition (displacement and rotation are specified
                         // i.e. all 6 DOFs are fixed)
    kRigidJoint = 3,     // Rigid constraint between two nodes (no relative motion between nodes i.e.
                         // all 6 DOFs of target node constrained)
    kRevoluteJoint = 4,  // Target node rotates freely around a specified axis (5 DOFs constrained)
    kRotationControl = 5,  // A rotation is specified about a given axis (1 DOF allowed about the
                           // axis specified by the user - the other 5 DOFs are constrained)
};

/// @brief Returns the number of nodes associated with a constraint type (1 or 2)
KOKKOS_INLINE_FUNCTION
constexpr size_t GetNumberOfNodes(ConstraintType type) {
    switch (type) {
        case ConstraintType::kRigidJoint:
        case ConstraintType::kRevoluteJoint:
        case ConstraintType::kRotationControl: {
            // kinematic pair: 2 nodes are required to specify the constraint relationship
            return 2U;
        } break;
        default:
            // kNone, kFixedBC, kPrescribedBC etc. require only one node to enforce the constraint
            return 1U;
    }
}

/// @brief Struct to define a constraint between two nodes or enforce a boundary condition at a
/// single node
/// @details A constraint is a relationship between two nodes that restricts their relative
/// motion in some way. Constraints can be used to model fixed boundary conditions, prescribed
/// displacements, rigid body motion, lower-pair kinematic joints, control signals etc.
struct Constraint {
    ConstraintType type;        //< Type of constraint
    size_t ID;                  //< Unique identifier for constraint
    Node base_node;             //< Base node for constraint
    Node target_node;           //< Target node for constraint
    Array_3 X0 = {0.};          //< Reference position for prescribed BC
    Array_3 x_axis = {0.};      //< Unit vector for x axis
    Array_3 y_axis = {0.};      //< Unit vector for y axis
    Array_3 z_axis = {0.};      //< Unit vector for z axis
    double* control = nullptr;  //< Pointer to control signal (if any)

    Constraint(
        ConstraintType constraint_type, size_t id, const Node& node1, const Node& node2,
        const Array_3& vec = {0., 0., 0.}, double* ctrl = nullptr
    )
        : type(constraint_type), ID(id), base_node(node1), target_node(node2), control(ctrl) {
        InitializeX0(vec);
        InitializeAxes(vec);
    }

    /// @brief Initializes X0 based on the constraint type and reference position
    void InitializeX0(const Array_3& vec) {
        // Set X0 to the prescribed displacement for fixed and prescribed BCs
        if (type == ConstraintType::kFixedBC || type == ConstraintType::kPrescribedBC) {
            X0[0] = target_node.x[0] - vec[0];
            X0[1] = target_node.x[1] - vec[1];
            X0[2] = target_node.x[2] - vec[2];
            return;
        }

        // Default: set X0 to the relative position between nodes
        X0[0] = target_node.x[0] - base_node.x[0];
        X0[1] = target_node.x[1] - base_node.x[1];
        X0[2] = target_node.x[2] - base_node.x[2];
    }

    /// @brief Initializes the x, y, z axes based on the constraint type
    void InitializeAxes(const Array_3& vec) {
        if (type == ConstraintType::kRevoluteJoint) {
            constexpr Array_3 x = {1., 0., 0.};
            const Array_3 x_hat = Norm(vec) > 0. ? UnitVector(vec) : UnitVector(X0);

            // Create rotation matrix to rotate x to match vector
            const auto v = CrossProduct(x, x_hat);
            const auto c = DotProduct(x_hat, x);
            const auto k = 1. / (1. + c);

            Array_3x3 R = {
                {{
                     v[0] * v[0] * k + c,
                     v[0] * v[1] * k - v[2],
                     v[0] * v[2] * k + v[1],
                 },
                 {
                     v[1] * v[0] * k + v[2],
                     v[1] * v[1] * k + c,
                     v[1] * v[2] * k - v[0],
                 },
                 {
                     v[2] * v[0] * k - v[1],
                     v[2] * v[1] * k + v[0],
                     v[2] * v[2] * k + c,
                 }}};

            // Set orthogonal unit vectors from the rotation matrix
            x_axis = {R[0][0], R[1][0], R[2][0]};
            y_axis = {R[0][1], R[1][1], R[2][1]};
            z_axis = {R[0][2], R[1][2], R[2][2]};
            return;
        }

        // If not a revolute/hinge joint, set axes to the input vector
        x_axis = vec;
    }

    /// @brief Returns the number of degrees of freedom used/fixed by the constraint
    [[nodiscard]] size_t NumDOFs() const {
        if (type == ConstraintType::kRevoluteJoint) {
            return 5U;  // A revolute joint constraints fixes 5 degrees of freedom
        }
        return static_cast<size_t>(kLieAlgebraComponents);  // Default: Fixes 6 DOFs
    }
};

}  // namespace openturbine
