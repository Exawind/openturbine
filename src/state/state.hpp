#pragma once

#include "dof_management/freedom_signature.hpp"

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
template <typename DeviceType>
struct State {
    size_t time_step{0};
    size_t num_system_nodes;
    Kokkos::View<size_t*, DeviceType> ID;
    Kokkos::View<FreedomSignature*, DeviceType> node_freedom_allocation_table;
    Kokkos::View<size_t*, DeviceType> active_dofs;
    Kokkos::View<size_t*, DeviceType> node_freedom_map_table;

    Kokkos::View<double* [7], DeviceType> x0;          //< Initial global position/rotation
    Kokkos::View<double* [7], DeviceType> x;           //< Current global position/rotation
    Kokkos::View<double* [6], DeviceType> q_delta;     //< Displacement increment
    Kokkos::View<double* [7], DeviceType> q_prev;      //< Previous state
    Kokkos::View<double* [7], DeviceType> q;           //< Current state
    Kokkos::View<double* [6], DeviceType> v;           //< Velocity
    Kokkos::View<double* [6], DeviceType> vd;          //< Acceleration
    Kokkos::View<double* [6], DeviceType> a;           //< Algorithmic acceleration
    Kokkos::View<double* [6], DeviceType> f;           //< External forces
    Kokkos::View<double* [6][6], DeviceType> tangent;  //< Tangent matrix

    explicit State(size_t num_system_nodes_)
        : num_system_nodes(num_system_nodes_),
          ID(Kokkos::view_alloc("ID", Kokkos::WithoutInitializing), num_system_nodes),
          node_freedom_allocation_table(
              Kokkos::view_alloc("node_freedom_allocation_table", Kokkos::WithoutInitializing),
              num_system_nodes
          ),
          active_dofs(
              Kokkos::view_alloc("active_dofs", Kokkos::WithoutInitializing), num_system_nodes
          ),
          node_freedom_map_table(
              Kokkos::view_alloc("node_freedom_map_table", Kokkos::WithoutInitializing),
              num_system_nodes
          ),
          x0(Kokkos::view_alloc("x0", Kokkos::WithoutInitializing), num_system_nodes),
          x(Kokkos::view_alloc("x", Kokkos::WithoutInitializing), num_system_nodes),
          q_delta(Kokkos::view_alloc("q_delta", Kokkos::WithoutInitializing), num_system_nodes),
          q_prev(Kokkos::view_alloc("q_prev", Kokkos::WithoutInitializing), num_system_nodes),
          q(Kokkos::view_alloc("q", Kokkos::WithoutInitializing), num_system_nodes),
          v(Kokkos::view_alloc("v", Kokkos::WithoutInitializing), num_system_nodes),
          vd(Kokkos::view_alloc("vd", Kokkos::WithoutInitializing), num_system_nodes),
          a(Kokkos::view_alloc("a", Kokkos::WithoutInitializing), num_system_nodes),
          f(Kokkos::view_alloc("f", Kokkos::WithoutInitializing), num_system_nodes),
          tangent(Kokkos::view_alloc("tangent", Kokkos::WithoutInitializing), num_system_nodes) {
        // Initialize q and q_prev rotation to identity
        Kokkos::deep_copy(Kokkos::subview(this->q_prev, Kokkos::ALL, 3), 1.);
        Kokkos::deep_copy(Kokkos::subview(this->q, Kokkos::ALL, 3), 1.);
    }
};

}  // namespace openturbine
