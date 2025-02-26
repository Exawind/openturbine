#pragma once

#include <Kokkos_Core.hpp>
#include <Kokkos_Profiling_ScopedRegion.hpp>

#include "constraints/constraints.hpp"

namespace openturbine {

inline void ResetConstraints(Constraints& constraints) {
    auto region = Kokkos::Profiling::ScopedRegion("Reset Constraints");
    Kokkos::deep_copy(constraints.lambda, 0.);
}

}  // namespace openturbine
