#pragma once

#include <Kokkos_Core.hpp>

namespace openturbine::springs {

template <typename DeviceType>
KOKKOS_INLINE_FUNCTION void CalculateForceVectors(
    const typename Kokkos::View<double[3], DeviceType>::const_type& r, double c1,
    const Kokkos::View<double[3], DeviceType>& f
) {
    for (auto component = 0U; component < 3U; ++component) {
        f(component) = c1 * r(component);
    }
}

}  // namespace openturbine::springs
