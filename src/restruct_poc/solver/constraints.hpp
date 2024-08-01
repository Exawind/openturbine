#pragma once

#include <array>
#include <vector>

#include <Kokkos_Core.hpp>

#include "constraint.hpp"

#include "src/restruct_poc/types.hpp"

namespace openturbine {

struct Constraints {
    struct DeviceData {
        ConstraintType type;
        Kokkos::pair<size_t, size_t> row_range;
        int base_node_index;
        int target_node_index;
        double X0[3];
        double axis_x[3];  // Unit vector representing x rotation axis
        double axis_y[3];  // Unit vector representing y rotation axis
        double axis_z[3];  // Unit vector representing z rotation axis
    };

    struct HostData {
        ConstraintType type;
        Array_7 u;
        double* control;
    };

    size_t num;
    size_t num_dofs;
    std::vector<HostData> constraint_data;
    Kokkos::View<DeviceData*> data;
    View_N control;
    View_Nx7 u;
    View_N Phi;
    View_NxN B;

    Constraints(const std::vector<Constraint>& constraints, size_t num_system_dofs)
        : num(constraints.size()),
          num_dofs(std::transform_reduce(
              constraints.cbegin(), constraints.cend(), 0U, std::plus{},
              [](auto c) {
                  return c.NumDOFs();
              }
          )),
          constraint_data(
              num, HostData{ConstraintType::kNone, {0., 0., 0., 1., 0., 0., 0.}, nullptr}
          ),
          data("data", num),
          control("control", num),
          u("u", num),
          Phi("residual_vector", num_dofs),
          B("gradient_matrix", num_dofs, num_system_dofs) {
        // Create host mirror for constraint data
        auto host_data = Kokkos::create_mirror(this->data);

        // Loop through constraint input
        auto start_row = size_t{0U};
        for (auto i = 0U; i < this->num; ++i) {
            // Set Host constraint data
            this->constraint_data[i].type = constraints[i].type;
            this->constraint_data[i].control = constraints[i].control;

            // Set constraint type
            host_data(i).type = constraints[i].type;

            // Set constraint rows
            auto dofs = constraints[i].NumDOFs();
            host_data(i).row_range = Kokkos::make_pair(start_row, start_row + dofs);
            start_row += dofs;

            // Set base node and target node index
            host_data(i).base_node_index = constraints[i].base_node.ID;
            host_data(i).target_node_index = constraints[i].target_node.ID;

            // Set initial relative location between nodes
            host_data(i).X0[0] = constraints[i].X0[0];
            host_data(i).X0[1] = constraints[i].X0[1];
            host_data(i).X0[2] = constraints[i].X0[2];

            // Set rotation axes
            for (auto j = 0U; j < 3U; ++j) {
                host_data(i).axis_x[j] = constraints[i].x_axis[j];
                host_data(i).axis_y[j] = constraints[i].y_axis[j];
                host_data(i).axis_z[j] = constraints[i].z_axis[j];
            }
        }

        // Update data
        Kokkos::deep_copy(this->data, host_data);

        // Set initial rotation to identity
        Kokkos::deep_copy(Kokkos::subview(this->u, Kokkos::ALL, 3), 1.0);
    }

    // UpdateDisplacement sets the new displacement for the given constraint
    void UpdateDisplacement(size_t id, Array_7 u_) { this->constraint_data[id].u = u_; }

    // UpdateViews transfers new prescribed displacements and control signals to views
    void UpdateViews() {
        // Prescribed displacement
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
