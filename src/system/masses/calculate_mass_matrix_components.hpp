#pragma once

#include <Kokkos_Core.hpp>

#include "math/vector_operations.hpp"

namespace openturbine::masses {

template <typename DeviceType>
KOKKOS_INLINE_FUNCTION void CalculateEta(
    const typename Kokkos::View<double[6][6], DeviceType>::const_type& Muu,
    const Kokkos::View<double[3], DeviceType>& eta
) {
    const auto m = Muu(0, 0);

    eta(0) = Muu(5, 1) / m;
    eta(1) = -Muu(5, 0) / m;
    eta(2) = Muu(4, 0) / m;
}

template <typename DeviceType>
KOKKOS_INLINE_FUNCTION void CalculateRho(
    const typename Kokkos::View<double[6][6], DeviceType>::const_type& Muu,
    const Kokkos::View<double[3][3], DeviceType>& rho
) {
    for (auto i = 0U; i < 3U; ++i) {
        for (auto j = 0U; j < 3U; ++j) {
            rho(i, j) = Muu(i + 3U, j + 3U);
        }
    }
}

}  // namespace openturbine::masses
