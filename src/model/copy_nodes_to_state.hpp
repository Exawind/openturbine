#pragma once

#include <memory>
#include <vector>

#include "node.hpp"
#include "state/state.hpp"
#include "state/update_global_position.hpp"

namespace openturbine {

template <typename DeviceType>
inline void CopyNodesToState(State<DeviceType>& state, const std::vector<Node>& nodes) {
    auto host_id = Kokkos::create_mirror_view(Kokkos::WithoutInitializing, state.ID);
    auto host_x0 = Kokkos::create_mirror_view(Kokkos::WithoutInitializing, state.x0);
    auto host_q = Kokkos::create_mirror_view(Kokkos::WithoutInitializing, state.q);
    auto host_v = Kokkos::create_mirror_view(Kokkos::WithoutInitializing, state.v);
    auto host_vd = Kokkos::create_mirror_view(Kokkos::WithoutInitializing, state.vd);

    for (auto i = 0U; i < nodes.size(); ++i) {
        const auto& node = nodes[i];
        for (auto j = 0U; j < kLieGroupComponents; ++j) {
            host_x0(i, j) = node.x0[j];
            host_q(i, j) = node.u[j];
        }
        for (auto j = 0U; j < kLieAlgebraComponents; ++j) {
            host_v(i, j) = node.v[j];
            host_vd(i, j) = node.vd[j];
        }
        host_id(i) = node.id;
    }

    // Copy data to host
    Kokkos::deep_copy(state.ID, host_id);
    Kokkos::deep_copy(state.x0, host_x0);
    Kokkos::deep_copy(state.q, host_q);
    Kokkos::deep_copy(state.v, host_v);
    Kokkos::deep_copy(state.vd, host_vd);
    Kokkos::deep_copy(state.a, state.vd);  // initialize algorithmic acceleration to acceleration
    Kokkos::deep_copy(state.f, 0.);

    // Set previous state to current state
    Kokkos::deep_copy(state.q_prev, state.q);

    // Calculate current global position from initial position and displacement
    Kokkos::parallel_for(
        "UpdateGlobalPosition",
        Kokkos::RangePolicy<typename DeviceType::execution_space>(0UL, state.num_system_nodes),
        UpdateGlobalPosition<DeviceType>{
            state.q,
            state.x0,
            state.x,
        }
    );
}

}  // namespace openturbine
