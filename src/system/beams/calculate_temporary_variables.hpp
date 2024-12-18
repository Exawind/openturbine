#pragma once

#include <KokkosBlas.hpp>
#include <Kokkos_Core.hpp>

#include "src/math/vector_operations.hpp"

namespace openturbine {

struct CalculateTemporaryVariables {
    size_t i_elem;
    Kokkos::View<double** [3]>::const_type qp_x0_prime_;
    Kokkos::View<double** [3]>::const_type qp_u_prime_;
    Kokkos::View<double** [3][3]> x0pupSS_;

    KOKKOS_FUNCTION
    void operator()(int i_qp) const {
        auto x0pup_data = Kokkos::Array<double, 3>{
            qp_x0_prime_(i_elem, i_qp, 0), qp_x0_prime_(i_elem, i_qp, 1),
            qp_x0_prime_(i_elem, i_qp, 2)
        };
        auto x0pup = View_3{x0pup_data.data()};
        auto u_prime_data = Kokkos::Array<double, 3>{
            qp_u_prime_(i_elem, i_qp, 0), qp_u_prime_(i_elem, i_qp, 1),
            qp_u_prime_(i_elem, i_qp, 2)
        };
        auto u_prime = View_3{u_prime_data.data()};
        KokkosBlas::serial_axpy(1., u_prime, x0pup);
        auto tmp_data = Kokkos::Array<double, 9>{};
        auto tmp = View_3x3{tmp_data.data()};
        VecTilde(x0pup, tmp);
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                x0pupSS_(i_elem, i_qp, i, j) = tmp(i, j);
            }
        }
    }
};

}  // namespace openturbine
