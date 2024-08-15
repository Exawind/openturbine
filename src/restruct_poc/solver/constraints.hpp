#pragma once

#include <array>
#include <numeric>
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
    /// @brief DeviceData struct holds constraint data on device
    struct DeviceData {
        ConstraintType type;                     //< Constraint type
        Kokkos::pair<size_t, size_t> row_range;  //< Range of rows in the global stiffness matrix
        Kokkos::pair<size_t, size_t> base_node_col_range;
        Kokkos::pair<size_t, size_t> target_node_col_range;
        size_t base_node_index;    //< Base node index
        size_t target_node_index;  //< Target node index
        double X0[3];              //< Initial relative location between nodes
        double axis_x[3];          // Unit vector representing x rotation axis
        double axis_y[3];          // Unit vector representing y rotation axis
        double axis_z[3];          // Unit vector representing z rotation axis
    };

    /// @brief HostData struct holds constraint data on host
    struct HostData {
        ConstraintType type;  //< Constraint type
        Array_7 u;            //< Prescribed displacement
        double* control;      //< Control signal
    };

    size_t num;                                    //< Number of constraints
    size_t num_dofs;                               //< Number of degrees of freedom
    std::vector<HostData> constraint_data;         //< Host constraint data
    Kokkos::View<DeviceData*> data;                //< Device constraint data
    View_N control;                                //< Control signals
    View_Nx7 u;                                    //< Prescribed displacements
    View_N Phi;                                    //< Residual vector
    Kokkos::View<double* [6][12]> gradient_terms;  //< Gradient terms

    explicit Constraints(const std::vector<std::shared_ptr<Constraint>>& constraints)
        : num{constraints.size()},
          num_dofs{std::transform_reduce(
              constraints.cbegin(), constraints.cend(), 0U, std::plus{},
              [](auto c) {
                  return c->NumDOFs();
              }
          )},
          constraint_data(
              num, HostData{ConstraintType::kNone, {0., 0., 0., 1., 0., 0., 0.}, nullptr}
          ),
          data("data", num),
          control("control", num),
          u("u", num),
          Phi("residual_vector", num_dofs),
          gradient_terms("gradient_terms", num) {
        // Create host mirror for constraint data
        auto host_data = Kokkos::create_mirror(this->data);

        // Loop through constraint input and set data
        auto start_row = size_t{0U};
        for (auto i = 0U; i < this->num; ++i) {
            // Set Host constraint data
            this->constraint_data[i].type = constraints[i]->type;
            this->constraint_data[i].control = constraints[i]->control;

            // Set constraint type
            host_data(i).type = constraints[i]->type;

            // Set constraint rows
            auto dofs = constraints[i]->NumDOFs();
            host_data(i).row_range = Kokkos::make_pair(start_row, start_row + dofs);
            start_row += dofs;

            if (GetNumberOfNodes(constraints[i]->type) == 2) {
                const auto target_node_id = constraints[i]->target_node.ID;
                const auto base_node_id = constraints[i]->base_node.ID;
                const auto target_start_col =
                    (target_node_id < base_node_id) ? 0U : kLieAlgebraComponents;
                host_data(i).target_node_col_range =
                    Kokkos::make_pair(target_start_col, target_start_col + kLieAlgebraComponents);

                const auto base_start_col =
                    (base_node_id < target_node_id) ? 0U : kLieAlgebraComponents;
                host_data(i).base_node_col_range =
                    Kokkos::make_pair(base_start_col, base_start_col + kLieAlgebraComponents);
            } else {
                host_data(i).target_node_col_range = Kokkos::make_pair(0U, kLieAlgebraComponents);
            }

            // Set base node and target node index
            host_data(i).base_node_index = constraints[i]->base_node.ID;
            host_data(i).target_node_index = constraints[i]->target_node.ID;

            // Set initial relative location between nodes
            host_data(i).X0[0] = constraints[i]->X0[0];
            host_data(i).X0[1] = constraints[i]->X0[1];
            host_data(i).X0[2] = constraints[i]->X0[2];

            // Set rotation axes
            for (auto j = 0U; j < 3U; ++j) {
                host_data(i).axis_x[j] = constraints[i]->x_axis[j];
                host_data(i).axis_y[j] = constraints[i]->y_axis[j];
                host_data(i).axis_z[j] = constraints[i]->z_axis[j];
            }
        }

        // Update data
        Kokkos::deep_copy(this->data, host_data);

        // Set initial rotation to identity
        Kokkos::deep_copy(Kokkos::subview(this->u, Kokkos::ALL, 3), 1.0);
    }

    /// Sets the new displacement for the given constraint
    void UpdateDisplacement(size_t id, Array_7 u_) { this->constraint_data[id].u = u_; }

    /// Transfers new prescribed displacements and control signals to Views
    void UpdateViews() {
        auto host_u_mirror = Kokkos::create_mirror(this->u);
        auto host_control_mirror = Kokkos::create_mirror(this->control);
        for (auto i = 0U; i < this->constraint_data.size(); ++i) {
            switch (this->constraint_data[i].type) {
                case ConstraintType::kPrescribedBC: {
                    for (auto j = 0U; j < static_cast<unsigned>(kLieGroupComponents); ++j) {
                        host_u_mirror(i, j) = this->constraint_data[i].u[j];
                    }
                } break;
                case ConstraintType::kRotationControl: {
                    host_control_mirror(i) = *this->constraint_data[i].control;
                } break;
                default:
                    break;
            }
        }
        Kokkos::deep_copy(this->control, host_control_mirror);
        Kokkos::deep_copy(this->u, host_u_mirror);
    }
};

}  // namespace openturbine
