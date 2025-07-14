#pragma once

#include <Kokkos_Core.hpp>

namespace openturbine::springs {

template <typename DeviceType>
KOKKOS_INLINE_FUNCTION void CalculateDistanceComponents(
    const typename Kokkos::View<double[3], DeviceType>::const_type& x0,
    const typename Kokkos::View<double[3], DeviceType>::const_type& u1,
    const typename Kokkos::View<double[3], DeviceType>::const_type& u2,
    const Kokkos::View<double[3], DeviceType>& r
) {
    for (auto component = 0U; component < 3U; ++component) {
        r(component) = x0(component) + u2(component) - u1(component);
    }
}

}  // namespace openturbine::springs
