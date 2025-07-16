#pragma once

#include <Kokkos_Core.hpp>

namespace openturbine::springs {

template <typename DeviceType>
KOKKOS_INLINE_FUNCTION double CalculateLength(
    const typename Kokkos::View<double[3], DeviceType>::const_type& r
) {
    return Kokkos::sqrt(r(0) * r(0) + r(1) * r(1) + r(2) * r(2));
}

}  // namespace openturbine::springs
