#pragma once

#include <vector>

#include "state.hpp"

namespace openturbine {

inline void CopyNodesToState(State& state, const std::vector<std::shared_ptr<Node>>& nodes) {
        auto host_q = Kokkos::create_mirror(state.q);
        auto host_v = Kokkos::create_mirror(state.v);
        auto host_vd = Kokkos::create_mirror(state.vd);

        for (auto i = 0U; i < nodes.size(); ++i) {
            const auto& node = *nodes[i];
            for (auto j = 0U; j < kLieGroupComponents; ++j) {
                host_q(i, j) = node.u[j];
            }
            for (auto j = 0U; j < kLieAlgebraComponents; ++j) {
                host_v(i, j) = node.v[j];
                host_vd(i, j) = node.vd[j];
            }
        }

        Kokkos::deep_copy(state.q, host_q);
        Kokkos::deep_copy(state.v, host_v);
        Kokkos::deep_copy(state.vd, host_vd);

        Kokkos::deep_copy(state.q_prev, state.q);
}

}
