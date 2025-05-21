#pragma once

#include "state.hpp"

namespace openturbine {

template <typename DeviceType>
inline State<DeviceType> CloneState(const State<DeviceType>& old) {
    auto clone = State<DeviceType>(old.num_system_nodes);
    Kokkos::deep_copy(clone.ID, old.ID);
    Kokkos::deep_copy(clone.node_freedom_allocation_table, old.node_freedom_allocation_table);
    Kokkos::deep_copy(clone.node_freedom_map_table, old.node_freedom_map_table);
    Kokkos::deep_copy(clone.x0, old.x0);
    Kokkos::deep_copy(clone.x, old.x);
    Kokkos::deep_copy(clone.q_delta, old.q_delta);
    Kokkos::deep_copy(clone.q_prev, old.q_prev);
    Kokkos::deep_copy(clone.q, old.q);
    Kokkos::deep_copy(clone.v, old.v);
    Kokkos::deep_copy(clone.vd, old.vd);
    Kokkos::deep_copy(clone.a, old.a);
    Kokkos::deep_copy(clone.f, old.f);
    Kokkos::deep_copy(clone.tangent, old.tangent);
    return clone;
}

}  // namespace openturbine
