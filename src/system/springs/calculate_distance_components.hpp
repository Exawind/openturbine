#pragma once

#include <KokkosBlas.hpp>
#include <Kokkos_Core.hpp>

#include "math/vector_operations.hpp"

namespace openturbine::springs {

template <typename DeviceType>
KOKKOS_INLINE_FUNCTION
void CalculateDistanceComponents(
    const typename Kokkos::View<double[3], DeviceType>::const_type& x0,
    const typename Kokkos::View<double[3], DeviceType>::const_type& u1,
    const typename Kokkos::View<double[3], DeviceType>::const_type& u2,
    const Kokkos::View<double[3], DeviceType>& r
) {
    for (auto i = 0U; i < 3U; ++i) {
        r(i) = x0(i) + u2(i) - u1(i);
    }
}

}  // namespace openturbine::springs
