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
};

}  // namespace openturbine
