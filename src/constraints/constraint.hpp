#pragma once

#include <cstddef>

#include "constraint_type.hpp"
#include "math/vector_operations.hpp"
#include "model/node.hpp"

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
    size_t ID;                       ///< Unique identifier for constraint
    ConstraintType type;             ///< Type of constraint
    std::array<size_t, 2> node_ids;  ///< Node IDs for: {base_node, target_node}
    Array_3 axis_vector;             ///< Vector for rotation/control axis
    Array_7 initial_displacement;    ///< Initial displacement for prescribed BC
    double* control;                 ///< Pointer to control signal (if any)

    /**
     * @brief Constructs a constraint with specified parameters
     *
     * @param id Unique identifier for the constraint
     * @param c_type Type of constraint
     * @param ids Array containing {base_node_id, target_node_id}
     * @param v Geometric configuration/axis vector (default: {0, 0, 0})
     * @param init_disp Initial displacement for prescribed BCs (default: {0, 0, 0, 1, 0, 0, 0})
     * @param ctrl Pointer to control signal (default: nullptr for static constraints)
     */
    Constraint(
        size_t id, ConstraintType c_type, const std::array<size_t, 2>& ids,
        const Array_3& v = {0., 0., 0.}, const Array_7& init_disp = {0., 0., 0., 1., 0., 0., 0.},
        double* ctrl = nullptr
    )
        : ID(id),
          type(c_type),
          node_ids(ids),
          axis_vector(v),
          initial_displacement(init_disp),
          control(ctrl) {}
};

}  // namespace openturbine
