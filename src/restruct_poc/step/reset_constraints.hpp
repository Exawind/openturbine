#pragma once

#include <Kokkos_Core.hpp>

#include "src/restruct_poc/constraints/constraints.hpp"

namespace openturbine {

inline void ResetConstraints(Constraints& constraints) {
    Kokkos::deep_copy(constraints.lambda, 0.);
}

}  // namespace openturbine
