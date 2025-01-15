#pragma once

#include "constraint_type.hpp"

#include "src/math/vector_operations.hpp"
#include "src/model/node.hpp"

namespace openturbine {

/// @brief Struct to define a constraint between two nodes or enforce a boundary condition at a
/// single node
/// @details A constraint is a relationship between two nodes that restricts their relative
/// motion in some way. Constraints can be used to model fixed boundary conditions, prescribed
/// displacements, rigid body motion, lower-pair kinematic joints, control signals etc.
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
