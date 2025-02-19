#pragma once

#include <KokkosBlas.hpp>
#include <Kokkos_Core.hpp>

#include "math/vector_operations.hpp"

namespace openturbine::beams {

KOKKOS_FUNCTION
inline void CalculateTemporaryVariables(
    const Kokkos::View<double[3]>::const_type& x0_prime,
    const Kokkos::View<double[3]>::const_type& u_prime, const Kokkos::View<double[3][3]>& x0pupSS
) {
    auto x0pup_data = Kokkos::Array<double, 3>{x0_prime(0), x0_prime(1), x0_prime(2)};
    const auto x0pup = Kokkos::View<double[3]>(x0pup_data.data());

    KokkosBlas::serial_axpy(1., u_prime, x0pup);
    VecTilde(x0pup, x0pupSS);
}

}  // namespace openturbine::beams
