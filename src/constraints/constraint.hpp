#pragma once

#include "constraint_type.hpp"

#include "src/math/vector_operations.hpp"
#include "src/model/node.hpp"

namespace openturbine {

/**
 * @brief Defines a constraint between two nodes or enforces a boundary condition at a single node
 *
 * @details A constraint establishes a relationship between two nodes, restricting their relative
 * motion in specific ways. This can be utilized to model various scenarios such as fixed boundary
 * conditions, prescribed displacements, rigid body motions, lower-pair kinematic joints, and control
 * signals.
 */
struct Constraint {
    ConstraintType type;        //< Type of constraint
    size_t ID;                  //< Unique identifier for constraint
    size_t base_node_id;        //< Base node for constraint
    size_t target_node_id;      //< Target node for constraint
    Array_3 vec = {0.};         //< Vector for initialization data
    double* control = nullptr;  //< Pointer to control signal (if any)

    Constraint(
        ConstraintType constraint_type, size_t id, const size_t node1_id, const size_t node2_id,
        const Array_3& v = {0., 0., 0.}, double* ctrl = nullptr
    )
        : type(constraint_type),
          ID(id),
          base_node_id(node1_id),
          target_node_id(node2_id),
          vec(v),
          control(ctrl) {}
};

}  // namespace openturbine
