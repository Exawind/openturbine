#pragma once

#include "src/restruct_poc/math/vector_operations.hpp"
#include "src/restruct_poc/model/node.hpp"

namespace openturbine {

/// @brief Enum class to define the type of constraint
enum class ConstraintType {
    kNone = 0,          // No constraint (default)
    kFixedBC = 1,       // Fixed boundary condition constraint (zero displacement)
    kPrescribedBC = 2,  // Prescribed boundary condition (displacement can be set)
    kRigid =
        3,  // Rigid constraint between two nodes (nodes maintain relative distance and rotation)
    kCylindrical = 4,      // Target node rotates freely around specified axis. Relative distance and
                           // rotation are fixed)
    kRotationControl = 5,  // Specify rotation about given axis
};

/// @brief Struct to define a constraint between two nodes
/// @details A constraint is a relationship between two nodes that restricts their relative motion
/// in some way. Constraints can be used to model fixed boundary conditions, prescribed
/// displacements, rigid body motion, and other types of constraints.
struct Constraint {
    ConstraintType type;       //< Type of constraint
    int ID;                    //< Unique identifier for constraint
    Node base_node;            //< Base node for constraint
    Node target_node;          //< Target node for constraint
    Array_3 X0 = {0.};         //< reference position for prescribed BC
    Array_3 x_axis = {0.};     //< unit vector for x axis
    Array_3 y_axis = {0.};     //< unit vector for y axis
    Array_3 z_axis = {0.};     //< unit vector for z axis
    float* control = nullptr;  //< Pointer to control signal

    Constraint(
        ConstraintType constraint_type, int id, const Node node1, const Node node2,
        Array_3 vec = {0., 0., 0.}, float* ctrl = nullptr
    )
        : type(constraint_type), ID(id), base_node(node1), target_node(node2), control(ctrl) {
        // If fixed BC or prescribed displacement, X0 is based on reference position vector
        if (constraint_type == ConstraintType::kFixedBC ||
            constraint_type == ConstraintType::kPrescribedBC) {
            this->X0[0] = this->target_node.x[0] - vec[0];
            this->X0[1] = this->target_node.x[1] - vec[1];
            this->X0[2] = this->target_node.x[2] - vec[2];
        } else {
            // Calculate initial difference in position between nodes
            this->X0[0] = this->target_node.x[0] - this->base_node.x[0];
            this->X0[1] = this->target_node.x[1] - this->base_node.x[1];
            this->X0[2] = this->target_node.x[2] - this->base_node.x[2];

            // If rotation control constraint, vec is rotation axis
            if (constraint_type == ConstraintType::kRotationControl) {
                this->x_axis = vec;
            } else if (constraint_type == ConstraintType::kCylindrical) {
                Array_3 x_hat = this->X0;
                auto length = sqrt(x_hat[0] * x_hat[0] + x_hat[1] * x_hat[1] + x_hat[2] * x_hat[2]);
                if (length > 1.0e-10) {
                    x_hat[0] /= length;
                    x_hat[1] /= length;
                    x_hat[2] /= length;
                } else {
                    x_hat = {1., 0., 0.};
                }
                Array_3 x = {1., 0., 0.};

                auto v = CrossProduct(x, x_hat);
                auto c = x[0] * x_hat[0] + x[1] * x_hat[1] + x[2] * x_hat[2];
                auto k = 1. / (1. + c);

                std::array<std::array<double, 3>, 3> R = {
                    {{v[0] * v[0] * k + c, v[0] * v[1] * k - v[2], v[0] * v[2] * k + v[1]},
                     {v[1] * v[0] * k + v[2], v[1] * v[1] * k + c, v[1] * v[2] * k - v[0]},
                     {v[2] * v[0] * k - v[1], v[2] * v[1] * k + v[0], v[2] * v[2] * k + c}}};

                this->x_axis = {R[0][0], R[1][0], R[2][0]};
                this->y_axis = {R[0][1], R[1][1], R[2][1]};
                this->z_axis = {R[0][2], R[1][2], R[2][2]};
            }
        }
    }

    /// Returns the number of degrees of freedom used by constraint
    int NumDOFs() const {
        switch (this->type) {
            case ConstraintType::kCylindrical: {
                return 5;
            } break;
            default: {
                return kLieAlgebraComponents;  // 7
            }
        }
    }
};

}  // namespace openturbine
