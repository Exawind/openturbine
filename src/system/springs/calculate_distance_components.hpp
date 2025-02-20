#pragma once

#include <KokkosBlas.hpp>
#include <Kokkos_Core.hpp>

#include "math/vector_operations.hpp"

namespace openturbine::springs {

KOKKOS_FUNCTION
inline void CalculateDistanceComponents(
    const Kokkos::View<double[3]>::const_type& x0, const Kokkos::View<double[3]>::const_type& u1,
    const Kokkos::View<double[3]>::const_type& u2, const Kokkos::View<double[3]>& r
) {
    for (auto i = 0U; i < 3U; ++i) {
        r(i) = x0(i) + u2(i) - u1(i);
    }
}

}  // namespace openturbine::springs
