#pragma once

#include <KokkosBlas.hpp>
#include <Kokkos_Core.hpp>

#include "math/vector_operations.hpp"

namespace openturbine::springs {

template <typename DeviceType>
KOKKOS_INLINE_FUNCTION
void CalculateForceVectors(
    const typename Kokkos::View<double[3], DeviceType>::const_type& r,
    double c1,
    const Kokkos::View<double[3], DeviceType>& f
) {
    for (auto i = 0U; i < 3U; ++i) {
        f(i) = c1 * r(i);
    }
}

}  // namespace openturbine::springs
