#pragma once

#include <KokkosBlas.hpp>
#include <Kokkos_Core.hpp>

#include "math/vector_operations.hpp"

namespace openturbine::springs {

KOKKOS_FUNCTION
inline void CalculateForceVectors(
    const Kokkos::View<double[3]>::const_type& r, double c1, const Kokkos::View<double[3]>& f
) {
    for (auto i = 0U; i < 3U; ++i) {
        f(i) = c1 * r(i);
    }
}

}  // namespace openturbine::springs
