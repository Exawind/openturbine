#pragma once

#include <KokkosBlas.hpp>
#include <Kokkos_Core.hpp>

#include "math/vector_operations.hpp"

namespace openturbine::beams {

KOKKOS_FUNCTION
inline void CalculateForceFC(
    const Kokkos::View<double[6][6]>::const_type& Cuu,
    const Kokkos::View<double[6]>::const_type& strain, const Kokkos::View<double[6]>& FC
) {
    using NoTranspose = KokkosBlas::Trans::NoTranspose;
    using Default = KokkosBlas::Algo::Gemv::Default;
    using Gemv = KokkosBlas::SerialGemv<NoTranspose, Default>;
    Gemv::invoke(1., Cuu, strain, 0., FC);
}

}  // namespace openturbine::beams
