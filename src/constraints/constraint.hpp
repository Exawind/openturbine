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
    size_t ID;                            //< Unique identifier for constraint
    ConstraintType type;                  //< Type of constraint
    std::array<size_t, 2> node_ids;       //< Node IDs for: {base_node, target_node}
    std::array<size_t, 2> node_num_dofs;  //< Number of DOFs associated w/: {base_node, target_node}
    Array_3 vec = {0.};                   //< Vector for initialization data
    double* control = nullptr;            //< Pointer to control signal (if any)

    Constraint(
        size_t id, ConstraintType constraint_type, const std::array<size_t, 2>& ids,
        const std::array<size_t, 2>& n_dofs = {6U, 6U}, const Array_3& v = {0., 0., 0.},
        double* ctrl = nullptr
    )
        : ID(id),
          type(constraint_type),
          node_ids(ids),
          node_num_dofs(n_dofs),
          vec(v),
          control(ctrl) {}
};

}  // namespace openturbine
