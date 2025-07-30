#pragma once

#include "state.hpp"

namespace openturbine {

template <typename DeviceType>
inline State<DeviceType> CloneState(const State<DeviceType>& old) {
    using Kokkos::deep_copy;

    auto clone = State<DeviceType>(old.num_system_nodes);
    clone.time_step = old.time_step;
    deep_copy(clone.ID, old.ID);
    deep_copy(clone.node_freedom_allocation_table, old.node_freedom_allocation_table);
    deep_copy(clone.node_freedom_map_table, old.node_freedom_map_table);
    deep_copy(clone.x0, old.x0);
    deep_copy(clone.x, old.x);
    deep_copy(clone.q_delta, old.q_delta);
    deep_copy(clone.q_prev, old.q_prev);
    deep_copy(clone.q, old.q);
    deep_copy(clone.v, old.v);
    deep_copy(clone.vd, old.vd);
    deep_copy(clone.a, old.a);
    deep_copy(clone.f, old.f);
    deep_copy(clone.tangent, old.tangent);
    return clone;
}

}  // namespace openturbine
