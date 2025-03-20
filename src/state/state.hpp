#pragma once

#include "dof_management/freedom_signature.hpp"
#include "types.hpp"

namespace openturbine {

/* @brief Container for storing the complete system state of the simulation at a given
 * time increment
 * @details This struct maintains all state variables required for the dynamic simulation,
 * including:
 * - Node configuration and freedom allocation
 * - Position and rotation states (current, previous, and initial)
 * - Velocity and acceleration states
 * - Tangent matrix
 */
struct State {
    size_t num_system_nodes;
    Kokkos::View<size_t*> ID;
    Kokkos::View<FreedomSignature*> node_freedom_allocation_table;
    Kokkos::View<size_t*> active_dofs;
    Kokkos::View<size_t*> node_freedom_map_table;

    View_Nx7 x0;                                   //< Initial global position/rotation
    View_Nx7 x;                                    //< Current global position/rotation
    View_Nx6 q_delta;                              //< Displacement increment
    View_Nx7 q_prev;                               //< Previous state
    View_Nx7 q;                                    //< Current state
    View_Nx6 v;                                    //< Velocity
    View_Nx6 vd;                                   //< Acceleration
    View_Nx6 a;                                    //< Algorithmic acceleration
    View_Nx6 f;                                    //< External forces
    Kokkos::View<double* [6]>::HostMirror host_f;  //< External forces mirror on host
    Kokkos::View<double* [6][6]> tangent;          //< Tangent matrix

    explicit State(size_t num_system_nodes_)
        : num_system_nodes(num_system_nodes_),
          ID("ID", num_system_nodes),
          node_freedom_allocation_table("node_freedom_allocation_table", num_system_nodes),
          active_dofs("active_dofs", num_system_nodes),
          node_freedom_map_table("node_freedom_map_table", num_system_nodes),
          x0("x0", num_system_nodes),
          x("x", num_system_nodes),
          q_delta("q_delta", num_system_nodes),
          q_prev("q_prev", num_system_nodes),
          q("q", num_system_nodes),
          v("v", num_system_nodes),
          vd("vd", num_system_nodes),
          a("a", num_system_nodes),
          f("f", num_system_nodes),
          host_f("host_f", num_system_nodes),
          tangent("tangent", num_system_nodes) {
        // Initialize q and q_prev rotation to identity
        Kokkos::deep_copy(Kokkos::subview(this->q_prev, Kokkos::ALL, 3), 1.);
        Kokkos::deep_copy(Kokkos::subview(this->q, Kokkos::ALL, 3), 1.);
    }
};

}  // namespace openturbine
