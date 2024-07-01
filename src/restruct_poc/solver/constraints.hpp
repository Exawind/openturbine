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
        int type;
        int updated;
    };

    int num_constraint_nodes;
    Kokkos::View<Data*> data;
    std::vector<double> control;
    std::vector<Array_7> host_u;
    View_Nx7 u;
    View_Nx3 X0;
    View_N Phi;
    View_NxN B;

    Constraints() {}
    Constraints(std::vector<Constraint> constraints, int num_system_nodes)
        : num_constraint_nodes(constraints.size()),
          data("data", num_constraint_nodes),
          control(num_constraint_nodes, 0.),
          host_u(num_constraint_nodes, {0., 0., 0., 1., 0., 0., 0.}),
          u("u", num_constraint_nodes),
          X0("X0", num_constraint_nodes),
          Phi("residual_vector", num_constraint_nodes * kLieAlgebraComponents),
          B("gradient_matrix", num_constraint_nodes * kLieAlgebraComponents,
            num_system_nodes * kLieAlgebraComponents) {
        // Create host mirror for constraint data
        auto host_data = Kokkos::create_mirror(this->data);
        auto host_X0 = Kokkos::create_mirror(this->X0);

        // Loop through constraint input
        for (size_t i = 0; i < constraints.size(); ++i) {
            host_data(i).base_node_index = constraints[i].base_node.ID;
            host_data(i).target_node_index = constraints[i].target_node.ID;
            host_data(i).type = constraints[i].type;

            host_X0(i, 0) = constraints[i].X0[0];
            host_X0(i, 1) = constraints[i].X0[1];
            host_X0(i, 2) = constraints[i].X0[2];
        }

        // Update data
        Kokkos::deep_copy(this->data, host_data);
        Kokkos::deep_copy(this->X0, host_X0);

        // Set initial rotation to identity
        Kokkos::deep_copy(Kokkos::subview(this->u, Kokkos::ALL, 3), 1.0);
    }

    void UpdateDisplacement(int index, Array_7 u_) { this->host_u[index] = u_; }

    void TransferU() {
        auto host_u_mirror = Kokkos::create_mirror(this->u);
        for (int i = 0; i < (int)this->host_u.size(); ++i) {
            for (int j = 0; j < kLieGroupComponents; ++j) {
                host_u_mirror(i, j) = this->host_u[i][j];
            }
        }
        Kokkos::deep_copy(this->u, host_u_mirror);
    }
};

}  // namespace openturbine
