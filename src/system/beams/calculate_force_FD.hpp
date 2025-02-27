#pragma once

#include <KokkosBlas.hpp>
#include <Kokkos_Core.hpp>

#include "math/vector_operations.hpp"

namespace openturbine::beams {

KOKKOS_FUNCTION
inline void CalculateForceFD(
    const Kokkos::View<double[3][3]>::const_type& x0pupSS,
    const Kokkos::View<double[6]>::const_type& FC, const Kokkos::View<double[6]>& FD
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
