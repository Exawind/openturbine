#pragma once

#include <array>
#include <vector>

#include <Kokkos_Core.hpp>

#include "ConstraintInput.hpp"

#include "src/restruct_poc/types.hpp"

namespace openturbine {

struct Constraints {
    struct NodeIndices {
        int base_node_index;
        int constrained_node_index;
    };

    int num_constraint_nodes;
    Kokkos::View<NodeIndices*> node_indices;
    View_N Phi;
    View_NxN B;
    View_Nx3 X0;
    View_Nx7 u;
    Constraints() {}
    Constraints(std::vector<ConstraintInput> inputs, int num_system_nodes)
        : num_constraint_nodes(inputs.size()),
          node_indices("node_indices", num_constraint_nodes),
          Phi("residual_vector", num_constraint_nodes * kLieAlgebraComponents),
          B("gradient_matrix", num_constraint_nodes * kLieAlgebraComponents,
            num_system_nodes * kLieAlgebraComponents),
          X0("X0", num_constraint_nodes),
          u("u", num_constraint_nodes) {
        auto host_node_indices = Kokkos::create_mirror(this->node_indices);
        for (size_t i = 0; i < inputs.size(); ++i) {
            host_node_indices(i).base_node_index = inputs[i].base_node_index;
            host_node_indices(i).constrained_node_index = inputs[i].constrained_node_index;
        }
        Kokkos::deep_copy(this->node_indices, host_node_indices);
        Kokkos::deep_copy(Kokkos::subview(this->u, Kokkos::ALL, 3), 1.0);
    }

    void UpdateDisplacement(int index, std::array<double, kLieGroupComponents> u_) {
        auto host_u = Kokkos::create_mirror(this->u);
        Kokkos::deep_copy(host_u, this->u);
        for (int i = 0; i < kLieGroupComponents; ++i) {
            host_u(index, i) = u_[i];
        }
        Kokkos::deep_copy(this->u, host_u);
    }
};

}  // namespace openturbine
