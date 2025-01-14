#pragma once

#include "state.hpp"

namespace openturbine {

inline void SetNodeExternalLoads(State& state, size_t i_node, const Array_6& loads) {
    auto host_f = Kokkos::create_mirror(state.f);
    Kokkos::deep_copy(host_f, state.f);
    for (auto i = 0U; i < kLieAlgebraComponents; ++i) {
        host_f(i_node, i) = loads[i];
    }
    Kokkos::deep_copy(state.f, host_f);
}

}  // namespace openturbine
