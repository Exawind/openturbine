#pragma once

#include <KokkosBlas.hpp>
#include <Kokkos_Core.hpp>

#include "math/vector_operations.hpp"

namespace openturbine::beams {

template <typename DeviceType>
KOKKOS_INLINE_FUNCTION void CalculateForceFC(
    const typename Kokkos::View<double[6][6], DeviceType>::const_type& Cuu,
    const typename Kokkos::View<double[6], DeviceType>::const_type& strain,
    const Kokkos::View<double[6], DeviceType>& FC
) {
    using NoTranspose = KokkosBlas::Trans::NoTranspose;
    using Default = KokkosBlas::Algo::Gemv::Default;
    using Gemv = KokkosBlas::SerialGemv<NoTranspose, Default>;
    Gemv::invoke(1., Cuu, strain, 0., FC);
}

}  // namespace openturbine::beams
