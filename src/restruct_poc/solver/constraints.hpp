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
        int base_node_index;
        int target_node_index;
        double X0[3];
        double axis[3];
    };

    struct HostData {
        ConstraintType type;
        Array_7 u;
        double* control;
    };

    int num_constraint_nodes;
    Kokkos::View<DeviceData*> data;
    std::vector<HostData> constraint_data;
    View_N control;
    View_Nx7 u;
    View_N Phi;
    View_NxN B;

    Constraints() {}
    Constraints(std::vector<Constraint> constraints, int num_system_nodes)
        : num_constraint_nodes(constraints.size()),
          data("data", num_constraint_nodes),
          constraint_data(
              num_constraint_nodes,
              HostData{ConstraintType::None, {0., 0., 0., 1., 0., 0., 0.}, nullptr}
          ),
          control("control", num_constraint_nodes),
          u("u", num_constraint_nodes),
          Phi("residual_vector", num_constraint_nodes * kLieAlgebraComponents),
          B("gradient_matrix", num_constraint_nodes * kLieAlgebraComponents,
            num_system_nodes * kLieAlgebraComponents) {
        // Create host mirror for constraint data
        auto host_data = Kokkos::create_mirror(this->data);

        // Loop through constraint input
        for (size_t i = 0; i < constraints.size(); ++i) {
            host_data(i).base_node_index = constraints[i].base_node.ID;
            host_data(i).target_node_index = constraints[i].target_node.ID;
            host_data(i).type = constraints[i].type;

            host_data(i).X0[0] = constraints[i].X0[0];
            host_data(i).X0[1] = constraints[i].X0[1];
            host_data(i).X0[2] = constraints[i].X0[2];

            host_data(i).axis[0] = constraints[i].rot_axis[0];
            host_data(i).axis[1] = constraints[i].rot_axis[1];
            host_data(i).axis[2] = constraints[i].rot_axis[2];

            this->constraint_data[i].type = constraints[i].type;
            this->constraint_data[i].control = constraints[i].control;
        }

        // Update data
        Kokkos::deep_copy(this->data, host_data);

        // Set initial rotation to identity
        Kokkos::deep_copy(Kokkos::subview(this->u, Kokkos::ALL, 3), 1.0);
    }

    // UpdateDisplacement sets the new displacement for the given constraint
    void UpdateDisplacement(int index, Array_7 u_) { this->constraint_data[index].u = u_; }

    // Transfer control signals and prescribed displacements to device
    void TransferToDevice() {
        // Prescribed displacement
        auto host_u_mirror = Kokkos::create_mirror(this->u);
        auto host_control_mirror = Kokkos::create_mirror(this->control);
        for (size_t i = 0; i < this->constraint_data.size(); ++i) {
            switch (this->constraint_data[i].type) {
                case ConstraintType::PrescribedBC:
                    for (int j = 0; j < kLieGroupComponents; ++j) {
                        host_u_mirror(i, j) = this->constraint_data[i].u[j];
                    }
                    break;
                case ConstraintType::RotationControl:
                    host_control_mirror(i) = *this->constraint_data[i].control;
                    break;
                default:
                    break;
            }
        }
        Kokkos::deep_copy(this->control, host_control_mirror);
        Kokkos::deep_copy(this->u, host_u_mirror);
    }
};

}  // namespace openturbine
