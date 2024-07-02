#pragma once

#include <array>
#include <vector>

#include <Kokkos_Core.hpp>

#include "constraint.hpp"

#include "src/restruct_poc/types.hpp"

namespace openturbine {

struct Constraints {
    struct Data {
        int base_node_index;
        int target_node_index;
        ConstraintType type;
        double X0[3];
        double axis[3];
    };

    int num_constraint_nodes;
    Kokkos::View<Data*> data;
    std::vector<double> host_control;
    std::vector<Array_7> host_u;
    View_N control;
    View_Nx7 u;
    View_N Phi;
    View_NxN B;

    Constraints() {}
    Constraints(std::vector<Constraint> constraints, int num_system_nodes)
        : num_constraint_nodes(constraints.size()),
          data("data", num_constraint_nodes),
          host_control(num_constraint_nodes, 0.),
          host_u(num_constraint_nodes, {0., 0., 0., 1., 0., 0., 0.}),
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
        }

        // Update data
        Kokkos::deep_copy(this->data, host_data);

        // Set initial rotation to identity
        Kokkos::deep_copy(Kokkos::subview(this->u, Kokkos::ALL, 3), 1.0);
    }

    // UpdateDisplacement sets the new displacement for the given constraint
    void UpdateDisplacement(int index, Array_7 u_) { this->host_u[index] = u_; }

    // UpdateControl sets the new control signal for the given constraint
    void UpdateControl(int index, double c) { this->host_control[index] = c; }

    // Transfer control signals and prescribed displacements to device
    void TransferToDevice() {
        // Prescribed displacement
        auto host_u_mirror = Kokkos::create_mirror(this->u);
        for (int i = 0; i < (int)this->host_u.size(); ++i) {
            for (int j = 0; j < kLieGroupComponents; ++j) {
                host_u_mirror(i, j) = this->host_u[i][j];
            }
        }
        Kokkos::deep_copy(this->u, host_u_mirror);

        // Control signals
        auto host_control_mirror = Kokkos::create_mirror(this->control);
        for (int i = 0; i < (int)this->host_control.size(); ++i) {
            host_control_mirror(i) = this->host_control[i];
        }
        Kokkos::deep_copy(this->control, host_control_mirror);
    }
};

}  // namespace openturbine
