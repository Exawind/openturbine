#pragma once

#include <KokkosBlas.hpp>
#include <Kokkos_Core.hpp>

#include "math/vector_operations.hpp"

namespace openturbine::beams {

template <typename DeviceType>
KOKKOS_INLINE_FUNCTION void CalculateForceFD(
    const typename Kokkos::View<double[3][3], DeviceType>::const_type& x0pupSS,
    const typename Kokkos::View<double[6], DeviceType>::const_type& FC,
    const Kokkos::View<double[6], DeviceType>& FD
) {
    using Transpose = KokkosBlas::Trans::Transpose;
    using Default = KokkosBlas::Algo::Gemv::Default;
    using Gemv = KokkosBlas::SerialGemv<Transpose, Default>;
    for (int i = 0; i < FD.extent_int(0); ++i) {
        FD(i) = 0.;
    }
    Gemv::invoke(
        1., x0pupSS, Kokkos::subview(FC, Kokkos::make_pair(0, 3)), 0.,
        Kokkos::subview(FD, Kokkos::make_pair(3, 6))
    );
}

}  // namespace openturbine::beams
