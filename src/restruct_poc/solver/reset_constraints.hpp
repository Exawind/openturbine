#pragma once

#include <Kokkos_Core.hpp>

#include "constraints.hpp"

namespace openturbine {

inline void ResetConstraints(Constraints& constraints) {
    Kokkos::deep_copy(constraints.lambda, 0.);
}

}
