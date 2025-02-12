#pragma once

#include <KokkosBlas.hpp>
#include <Kokkos_Core.hpp>

namespace openturbine::masses {

KOKKOS_FUNCTION
inline void CalculateGravityForce(double mass, const Kokkos::View<double[3]>::const_type& gravity, const Kokkos::View<double[3][3]>::const_type& eta_tilde, const Kokkos::View<double[6]>& FG) {
        using NoTranspose = KokkosBlas::Trans::NoTranspose;
        using Default = KokkosBlas::Algo::Gemv::Default;
        using Gemv = KokkosBlas::SerialGemv<NoTranspose, Default>;

        KokkosBlas::serial_axpy(mass, gravity, Kokkos::subview(FG, Kokkos::make_pair(0, 3)));
        Gemv::invoke(1., eta_tilde, Kokkos::subview(FG, Kokkos::make_pair(0, 3)), 0., Kokkos::subview(FG, Kokkos::make_pair(3, 6)));
}

}  // namespace openturbine::masses
