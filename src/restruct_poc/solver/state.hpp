#pragma once

#include "src/restruct_poc/types.hpp"

namespace openturbine {

/// @brief State struct holds all state data
/// @details State struct holds all state data and provides methods to update the
/// state variables during the simulation
struct State {
    size_t num_system_nodes;     //< Number of system nodes
    size_t num_constraint_dofs;  //< Number of constraint degrees of freedom
    View_Nx6 q_delta;            //< Displacement increment
    View_Nx7 q_prev;             //< Previous state
    View_Nx7 q;                  //< Current state
    View_Nx6 v;                  //< Velocity
    View_Nx6 vd;                 //< Acceleration
    View_Nx6 a;                  //< Algorithmic acceleration
    View_N lambda;               //< Lagrange multipliers

    State(size_t num_system_nodes_, size_t num_constraint_dofs_)
        : num_system_nodes(num_system_nodes_),
          num_constraint_dofs(num_constraint_dofs_),
          q_delta("q_delta", num_system_nodes),
          q_prev("q_prev", num_system_nodes),
          q("q", num_system_nodes),
          v("v", num_system_nodes),
          vd("vd", num_system_nodes),
          a("a", num_system_nodes),
          lambda("lambda", num_constraint_dofs) {
        // Initialize q and q_prev rotation to identity
        Kokkos::deep_copy(Kokkos::subview(this->q_prev, Kokkos::ALL, 3), 1.);
        Kokkos::deep_copy(Kokkos::subview(this->q, Kokkos::ALL, 3), 1.);
    }

    State(
        size_t num_system_nodes_, size_t num_constraint_dofs_,
        const std::vector<std::shared_ptr<Node>>& nodes
    )
        : State(num_system_nodes_, num_constraint_dofs_) {
        // Create mirror of state views
        auto host_q = Kokkos::create_mirror(this->q);
        auto host_v = Kokkos::create_mirror(this->v);
        auto host_vd = Kokkos::create_mirror(this->vd);

        // Loop through number of nodes and copy data to host view
        for (auto i = 0U; i < nodes.size(); ++i) {
            const auto& node = nodes[i];
            for (auto j = 0U; j < kLieGroupComponents; ++j) {
                host_q(i, j) = node->u[j];
            }
            for (auto j = 0U; j < kLieAlgebraComponents; ++j) {
                host_v(i, j) = node->v[j];
                host_vd(i, j) = node->vd[j];
            }
        }

        // Transfer host view data to state views
        Kokkos::deep_copy(this->q, host_q);
        Kokkos::deep_copy(this->v, host_v);
        Kokkos::deep_copy(this->vd, host_vd);

        // Copy q to q_previous
        Kokkos::deep_copy(this->q_prev, this->q);
    }
};

}  // namespace openturbine
