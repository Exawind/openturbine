#pragma once

#include "state.hpp"

namespace openturbine {

inline void CopyStateData(State& copy, const State& old) {
    Kokkos::deep_copy(copy.x, old.x);
    Kokkos::deep_copy(copy.q_delta, old.q_delta);
    Kokkos::deep_copy(copy.q_prev, old.q_prev);
    Kokkos::deep_copy(copy.q, old.q);
    Kokkos::deep_copy(copy.v, old.v);
    Kokkos::deep_copy(copy.vd, old.vd);
    Kokkos::deep_copy(copy.a, old.a);
    Kokkos::deep_copy(copy.tangent, old.tangent);
}

}  // namespace openturbine
