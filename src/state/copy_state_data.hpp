#pragma once

#include "state.hpp"

namespace openturbine {

template <typename DeviceType>
inline void CopyStateData(State<DeviceType>& copy, const State<DeviceType>& old) {
    using Kokkos::deep_copy;

    copy.time_step = old.time_step;
    deep_copy(copy.x, old.x);
    deep_copy(copy.q_delta, old.q_delta);
    deep_copy(copy.q_prev, old.q_prev);
    deep_copy(copy.q, old.q);
    deep_copy(copy.v, old.v);
    deep_copy(copy.vd, old.vd);
    deep_copy(copy.a, old.a);
    deep_copy(copy.f, old.f);
    deep_copy(copy.tangent, old.tangent);
}

}  // namespace openturbine
