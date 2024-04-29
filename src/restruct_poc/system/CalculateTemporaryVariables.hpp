#pragma once

#include <Kokkos_Core.hpp>

#include "src/restruct_poc/VectorOperations.hpp"
#include "src/restruct_poc/types.hpp"

namespace openturbine {

struct CalculateTemporaryVariables {
    View_Nx3::const_type qp_x0_prime_;
    View_Nx3::const_type qp_u_prime_;
    View_Nx3x3 x0pupSS_;

    KOKKOS_FUNCTION
    void operator()(int i_qp) const {
        auto x0pup_data = Kokkos::Array<double, 3>{
            qp_x0_prime_(i_qp, 0), qp_x0_prime_(i_qp, 1), qp_x0_prime_(i_qp, 2)};
        auto x0pup =
            Kokkos::View<double[3], Kokkos::MemoryTraits<Kokkos::Unmanaged>>{x0pup_data.data()};
        auto u_prime_data = Kokkos::Array<double, 3>{
            qp_u_prime_(i_qp, 0), qp_u_prime_(i_qp, 1), qp_u_prime_(i_qp, 2)};
        auto u_prime =
            Kokkos::View<double[3], Kokkos::MemoryTraits<Kokkos::Unmanaged>>{u_prime_data.data()};
        KokkosBlas::serial_axpy(1., u_prime, x0pup);
        auto tmp_data = Kokkos::Array<double, 9>{};
        auto tmp =
            Kokkos::View<double[3][3], Kokkos::MemoryTraits<Kokkos::Unmanaged>>{tmp_data.data()};
        VecTilde(x0pup, tmp);
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                x0pupSS_(i_qp, i, j) = tmp(i, j);
            }
        }
    }
};

}  // namespace openturbine
