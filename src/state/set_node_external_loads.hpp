#pragma once

#include "state.hpp"

namespace openturbine {

inline void SetNodeExternalLoads(State& state, size_t i_node, const Array_6& loads) {
    for (auto i = 0U; i < kLieAlgebraComponents; ++i) {
        state.host_f(i_node, i) = loads[i];
    }
}

}  // namespace openturbine
