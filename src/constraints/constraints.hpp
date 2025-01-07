#pragma once

#include <array>
#include <numeric>
#include <vector>

#include <Kokkos_Core.hpp>

#include "constraint.hpp"

#include "src/dof_management/freedom_signature.hpp"

namespace openturbine {

/**
 * @brief Container class for managing multiple constraints in a simulation
 *
 * @details Manages a collection of constraints between nodes, including their properties,
 * freedom signatures, and computational data structures. This class handles both single-node
 * boundary conditions and two-node constraints (like joints). It provides facilities for:
 * - Storing constraint properties (types, node indices, axes)
 * - Managing degrees of freedom and freedom signatures
 * - Handling control signals and prescribed displacements
 * - Maintaining computational views for residuals and gradients
 */
struct Constraints {
    size_t num_constraints;  //< Total number of constraints
    size_t num_dofs;         //< Total number of degrees of freedom

    // Constraint properties
    std::vector<double*> control_signal;      //< Control signal for each constraint
    Kokkos::View<ConstraintType*> type;       //< Type of each constraint
    Kokkos::View<size_t*> base_node_index;    //< Index of the base node for each constraint
    Kokkos::View<size_t*> target_node_index;  //< Index of the target node for each constraint

    // DOF management
    Kokkos::View<Kokkos::pair<size_t, size_t>*> row_range;
    Kokkos::View<Kokkos::pair<size_t, size_t>*> base_node_col_range;
    Kokkos::View<Kokkos::pair<size_t, size_t>*> target_node_col_range;
    Kokkos::View<FreedomSignature*> base_node_freedom_signature;
    Kokkos::View<FreedomSignature*> target_node_freedom_signature;
    Kokkos::View<size_t* [6]> base_node_freedom_table;
    Kokkos::View<size_t* [6]> target_node_freedom_table;

    // Geometric configuration
    Kokkos::View<double* [3]> X0;       //< Initial relative position
    Kokkos::View<double* [3][3]> axes;  //< Rotation axes

    // State variables
    Kokkos::View<double* [7]> input;   //< Current state input
    Kokkos::View<double* [3]> output;  //< Current state output
    Kokkos::View<double*> lambda;      //< Lagrange multipliers

    // Host mirrors for CPU access
    Kokkos::View<double* [7]>::HostMirror host_input;
    Kokkos::View<double* [3]>::HostMirror host_output;

    // System contributions
    Kokkos::View<double* [6]> residual_terms;
    Kokkos::View<double* [6]> system_residual_terms;
    Kokkos::View<double* [6][6]> base_gradient_terms;
    Kokkos::View<double* [6][6]> target_gradient_terms;

    explicit Constraints(const std::vector<std::shared_ptr<Constraint>>& constraints)
        : num_constraints{constraints.size()},
          num_dofs{std::transform_reduce(
              constraints.cbegin(), constraints.cend(), 0U, std::plus{},
              [](auto c) {
                  return NumDOFsForConstraint(c->type);
              }
          )},
          control_signal(num_constraints),
          type("type", num_constraints),
          base_node_index("base_node_index", num_constraints),
          target_node_index("target_node_index", num_constraints),
          row_range("row_range", num_constraints),
          base_node_col_range("base_row_col_range", num_constraints),
          target_node_col_range("target_row_col_range", num_constraints),
          base_node_freedom_signature("base_node_freedom_signature", num_constraints),
          target_node_freedom_signature("target_node_freedom_signature", num_constraints),
          base_node_freedom_table("base_node_freedom_table", num_constraints),
          target_node_freedom_table("target_node_freedom_table", num_constraints),
          X0("X0", num_constraints),
          axes("axes", num_constraints),
          input("inputs", num_constraints),
          output("outputs", num_constraints),
          lambda("lambda", num_dofs),
          host_input("host_input", num_constraints),
          host_output("host_output", num_constraints),
          residual_terms("residual_terms", num_constraints),
          system_residual_terms("system_residual_terms", num_constraints),
          base_gradient_terms("base_gradient_terms", num_constraints),
          target_gradient_terms("target_gradient_terms", num_constraints) {
        Kokkos::deep_copy(base_node_freedom_signature, FreedomSignature::AllComponents);
        Kokkos::deep_copy(target_node_freedom_signature, FreedomSignature::AllComponents);

        auto host_type = Kokkos::create_mirror(type);
        auto host_row_range = Kokkos::create_mirror(row_range);
        auto host_base_node_col_range = Kokkos::create_mirror(base_node_col_range);
        auto host_target_node_col_range = Kokkos::create_mirror(target_node_col_range);
        auto host_base_node_index = Kokkos::create_mirror(base_node_index);
        auto host_target_node_index = Kokkos::create_mirror(target_node_index);
        auto host_X0 = Kokkos::create_mirror(X0);
        auto host_axes = Kokkos::create_mirror(axes);

        auto start_row = size_t{0U};
        for (auto i = 0U; i < num_constraints; ++i) {
            host_type(i) = constraints[i]->type;
            control_signal[i] = constraints[i]->control;

            auto dofs = NumDOFsForConstraint(host_type(i));
            host_row_range(i) = Kokkos::make_pair(start_row, start_row + dofs);
            start_row += dofs;

            if (GetNumberOfNodes(constraints[i]->type) == 2) {
                const auto target_node_id = constraints[i]->target_node.ID;
                const auto base_node_id = constraints[i]->base_node.ID;
                const auto target_start_col =
                    (target_node_id < base_node_id) ? 0U : kLieAlgebraComponents;
                host_target_node_col_range(i) =
                    Kokkos::make_pair(target_start_col, target_start_col + kLieAlgebraComponents);

                const auto base_start_col =
                    (base_node_id < target_node_id) ? 0U : kLieAlgebraComponents;
                host_base_node_col_range(i) =
                    Kokkos::make_pair(base_start_col, base_start_col + kLieAlgebraComponents);
            } else {
                host_target_node_col_range(i) = Kokkos::make_pair(0U, kLieAlgebraComponents);
            }

            host_base_node_index(i) = constraints[i]->base_node.ID;
            host_target_node_index(i) = constraints[i]->target_node.ID;

            // Set initial relative location between nodes and rotation axes
            for (auto j = 0U; j < 3U; ++j) {
                host_X0(i, j) = constraints[i]->X0[j];
                host_axes(i, 0, j) = constraints[i]->x_axis[j];
                host_axes(i, 1, j) = constraints[i]->y_axis[j];
                host_axes(i, 2, j) = constraints[i]->z_axis[j];
            }
        }

        Kokkos::deep_copy(type, host_type);
        Kokkos::deep_copy(row_range, host_row_range);
        Kokkos::deep_copy(base_node_col_range, host_base_node_col_range);
        Kokkos::deep_copy(target_node_col_range, host_target_node_col_range);
        Kokkos::deep_copy(base_node_index, host_base_node_index);
        Kokkos::deep_copy(target_node_index, host_target_node_index);
        Kokkos::deep_copy(X0, host_X0);
        Kokkos::deep_copy(axes, host_axes);

        Kokkos::deep_copy(Kokkos::subview(this->input, Kokkos::ALL, 3), 1.);
    }

    /// Sets the new displacement for the given constraint
    void UpdateDisplacement(size_t constraint_id, const std::array<double, 7>& disp) const {
        for (auto i = 0U; i < 7U; ++i) {
            host_input(constraint_id, i) = disp[i];
        }
    }

    /// Transfers new prescribed displacements and control signals to Views
    void UpdateViews() {
        for (auto i = 0U; i < this->num_constraints; ++i) {
            if (control_signal[i] != nullptr) {
                host_input(i, 0) = *control_signal[i];
            }
        }
        Kokkos::deep_copy(this->input, host_input);
    }
};

}  // namespace openturbine
