#pragma once

#include <array>
#include <vector>

#include <Kokkos_Core.hpp>

#include "constraint_input.hpp"

#include "src/restruct_poc/types.hpp"

namespace openturbine {

struct Constraints {
    struct Data {
        int base_node_index;
        int target_node_index;
        int type;
        double X0[3];
    };

    int num_constraint_nodes;
    Kokkos::View<Data*> data;
    View_Nx7 control;
    View_N Phi;
    View_NxN B;

    Constraints() {}
    Constraints(std::vector<ConstraintInput> inputs, int num_system_nodes)
        : num_constraint_nodes(inputs.size()),
          data("data", num_constraint_nodes),
          control("control", num_constraint_nodes),
          Phi("residual_vector", num_constraint_nodes * kLieAlgebraComponents),
          B("gradient_matrix", num_constraint_nodes * kLieAlgebraComponents,
            num_system_nodes * kLieAlgebraComponents) {
        // Create host mirror for constraint data
        auto host_data = Kokkos::create_mirror(this->data);

        // Loop through constraint input
        for (size_t i = 0; i < inputs.size(); ++i) {
            host_data(i).base_node_index = inputs[i].base_node_index;
            host_data(i).target_node_index = inputs[i].target_node_index;
            host_data(i).type = inputs[i].type;
            host_data(i).X0[0] = inputs[i].x0[0];
            host_data(i).X0[1] = inputs[i].x0[1];
            host_data(i).X0[2] = inputs[i].x0[2];
        }

        // Update data
        Kokkos::deep_copy(this->data, host_data);

        // Set initial rotation to identity
        Kokkos::deep_copy(Kokkos::subview(this->control, Kokkos::ALL, 3), 1.0);
    }

    void UpdateDisplacement(int index, std::array<double, kLieGroupComponents> u_) {
        auto host_control = Kokkos::create_mirror(this->control);
        Kokkos::deep_copy(host_control, this->control);
        for (int i = 0; i < kLieGroupComponents; ++i) {
            host_control(index, i) = u_[i];
        }
        Kokkos::deep_copy(this->control, host_control);
    }
};

}  // namespace openturbine
