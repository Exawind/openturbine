#pragma once

#include <array>
#include <vector>

#include <Kokkos_Core.hpp>

#include "constraint.hpp"

#include "src/restruct_poc/types.hpp"

namespace openturbine {

/// @brief Constraints struct holds all constraint data
/// @details Constraints struct holds all constraint data and provides methods
/// to update the prescribed displacements and control signals.
/// @note The struct is used to transfer data between the solver and the
/// constraint objects.
struct Constraints {
    size_t num;       //< Number of constraints
    size_t num_dofs;  //< Number of degrees of freedom
    std::vector<double*> control_signal;
    Kokkos::View<ConstraintType*> type;
    Kokkos::View<Kokkos::pair<size_t, size_t>*> row_range;
    Kokkos::View<Kokkos::pair<size_t, size_t>*> base_node_col_range;
    Kokkos::View<Kokkos::pair<size_t, size_t>*> target_node_col_range;
    Kokkos::View<size_t*> base_node_index;
    Kokkos::View<size_t*> target_node_index;
    Kokkos::View<double* [3]> X0;
    Kokkos::View<double* [3][3]> axes;

    View_N control;  //< Control signals
    View_Nx7 u;      //< Prescribed displacements
    Kokkos::View<double* [7]>::HostMirror u_signal;
    Kokkos::View<double*> lambda;
    Kokkos::View<double* [6]> residual_terms;
    Kokkos::View<double* [6][6]> base_gradient_terms;
    Kokkos::View<double* [6][6]> target_gradient_terms;

    Constraints(const std::vector<std::shared_ptr<Constraint>>& constraints)
        : num{constraints.size()},
          num_dofs{std::transform_reduce(
              constraints.cbegin(), constraints.cend(), 0U, std::plus{},
              [](auto c) {
                  return c->NumDOFs();
              }
          )},
          control_signal(num),
          type("type", num),
          row_range("row_range", num),
          base_node_col_range("base_row_col_range", num),
          target_node_col_range("target_row_col_range", num),
          base_node_index("base_node_index", num),
          target_node_index("target_node_index", num),
          X0("X0", num),
          axes("axes", num),
          control("control", num),
          u("u", num),
          u_signal("u_signal", num),
          lambda("lambda", num_dofs),
          residual_terms("residual_terms", num),
          base_gradient_terms("base_gradient_terms", num),
          target_gradient_terms("target_gradient_terms", num) {
        // Create host mirror for constraint data
        auto host_type = Kokkos::create_mirror(type);
        auto host_row_range = Kokkos::create_mirror(row_range);
        auto host_base_node_col_range = Kokkos::create_mirror(base_node_col_range);
        auto host_target_node_col_range = Kokkos::create_mirror(target_node_col_range);
        auto host_base_node_index = Kokkos::create_mirror(base_node_index);
        auto host_target_node_index = Kokkos::create_mirror(target_node_index);
        auto host_X0 = Kokkos::create_mirror(X0);
        auto host_axes = Kokkos::create_mirror(axes);

        // Loop through constraint input and set data
        auto start_row = size_t{0U};
        for (auto i = 0U; i < num; ++i) {
            // Set Host constraint data
            host_type(i) = constraints[i]->type;
            control_signal[i] = constraints[i]->control;

            // Set constraint rows
            auto dofs = constraints[i]->NumDOFs();
            host_row_range(i) = Kokkos::make_pair(start_row, start_row + dofs);
            start_row += dofs;

            if (GetNumberOfNodes(constraints[i]->type) == 2) {
                const auto target_node_id = constraints[i]->target_node.ID;
                const auto base_node_id = constraints[i]->base_node.ID;
                const auto target_start_col =
                    (target_node_id < base_node_id) ? 0U : kLieAlgebraComponents;
                host_target_node_col_range(i) = Kokkos::make_pair(target_start_col, target_start_col + kLieAlgebraComponents);

                const auto base_start_col =
                    (base_node_id < target_node_id) ? 0U : kLieAlgebraComponents;
                host_base_node_col_range(i) = Kokkos::make_pair(base_start_col, base_start_col + kLieAlgebraComponents);
            } else {
                host_target_node_col_range(i) = Kokkos::make_pair(0U, kLieAlgebraComponents);
            }

            // Set base node and target node index
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

        // Update data
        Kokkos::deep_copy(type, host_type);
        Kokkos::deep_copy(row_range, host_row_range);
        Kokkos::deep_copy(base_node_col_range, host_base_node_col_range);
        Kokkos::deep_copy(target_node_col_range, host_target_node_col_range);
	Kokkos::deep_copy(base_node_index, host_base_node_index);
        Kokkos::deep_copy(target_node_index, host_target_node_index);
        Kokkos::deep_copy(X0, host_X0);
        Kokkos::deep_copy(axes, host_axes);

        // Set initial rotation to identity
        Kokkos::deep_copy(Kokkos::subview(this->u, Kokkos::ALL, 3), 1.0);
    }

    /// Sets the new displacement for the given constraint
    void UpdateDisplacement(size_t id, const std::array<double, 7>& u_) const {
        for (auto i = 0U; i < 7U; ++i) {
            u_signal(id, i) = u_[i];
        }
    }

    /// Transfers new prescribed displacements and control signals to Views
    void UpdateViews() {
        auto host_control_mirror = Kokkos::create_mirror(this->control);
        auto host_type = Kokkos::create_mirror(type);
        Kokkos::deep_copy(host_type, type);
        for (auto i = 0U; i < num; ++i) {
            switch (host_type(i)) {
                case ConstraintType::kRotationControl: {
                    host_control_mirror(i) = *control_signal[i];
                } break;
                default:
                    break;
            }
        }
        Kokkos::deep_copy(this->control, host_control_mirror);
        Kokkos::deep_copy(this->u, u_signal);
    }
};

}  // namespace openturbine
