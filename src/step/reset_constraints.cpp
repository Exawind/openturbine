#include <Kokkos_Core.hpp>

#include "src/constraints/constraints.hpp"

namespace openturbine {

void ResetConstraints(const Constraints& constraints) {
    Kokkos::deep_copy(constraints.lambda, 0.);
}

}  // namespace openturbine
