#pragma once

#include <KokkosBatched_Gemv_Decl.hpp>
#include <KokkosBlas.hpp>
#include <Kokkos_Core.hpp>

#include "math/quaternion_operations.hpp"

namespace openturbine::beams {
template <typename DeviceType>
KOKKOS_INLINE_FUNCTION
void CalculateStrain(
    const typename Kokkos::View<double[3], DeviceType>::const_type& x0_prime,
    const typename Kokkos::View<double[3], DeviceType>::const_type& u_prime,
    const typename Kokkos::View<double[4], DeviceType>::const_type& r,
    const typename Kokkos::View<double[4], DeviceType>::const_type& r_prime,
    const Kokkos::View<double[6], DeviceType>& strain
) {
    using NoTranspose = KokkosBlas::Trans::NoTranspose;
    using Default = KokkosBlas::Algo::Gemv::Default;
    using Gemv = KokkosBlas::SerialGemv<NoTranspose, Default>;
    auto R_x0_prime_data = Kokkos::Array<double, 3>{};
    auto R_x0_prime = Kokkos::View<double[3]>(R_x0_prime_data.data());

    RotateVectorByQuaternion(r, x0_prime, R_x0_prime);
    KokkosBlas::serial_axpy(-1., u_prime, R_x0_prime);
    KokkosBlas::serial_axpy(-1., x0_prime, R_x0_prime);

    auto E_data = Kokkos::Array<double, 12>{};
    auto E = Kokkos::View<double[3][4]>(E_data.data());
    QuaternionDerivative(r, E);
    auto e2_data = Kokkos::Array<double, 4>{};
    auto e2 = Kokkos::View<double[4]>{e2_data.data()};
    Gemv::invoke(2., E, r_prime, 0., e2);

    strain(0) = -R_x0_prime(0);
    strain(1) = -R_x0_prime(1);
    strain(2) = -R_x0_prime(2);
    strain(3) = e2(0);
    strain(4) = e2(1);
    strain(5) = e2(2);
}
}  // namespace openturbine::beams
