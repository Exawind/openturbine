#pragma once

#include <Kokkos_Core.hpp>

#include "src/restruct_poc/math/vector_operations.hpp"
#include "src/restruct_poc/types.hpp"

namespace openturbine {

struct CalculateMassMatrixComponents {
    size_t i_elem;
    Kokkos::View<double** [6][6]>::const_type qp_Muu_;
    Kokkos::View<double** [3]> eta_;
    Kokkos::View<double** [3][3]> rho_;
    Kokkos::View<double** [3][3]> eta_tilde_;

    KOKKOS_FUNCTION
    void operator()(int i_qp) const {
        auto Muu = Kokkos::subview(qp_Muu_, i_elem, i_qp, Kokkos::ALL, Kokkos::ALL);
        auto eta = Kokkos::subview(eta_, i_elem, i_qp, Kokkos::ALL);
        auto rho = Kokkos::subview(rho_, i_elem, i_qp, Kokkos::ALL, Kokkos::ALL);
        auto eta_tilde = Kokkos::subview(eta_tilde_, i_elem, i_qp, Kokkos::ALL, Kokkos::ALL);

        auto m = Muu(0, 0);
        if (m == 0.) {
            eta(0) = 0.;
            eta(1) = 0.;
            eta(2) = 0.;
        } else {
            eta(0) = Muu(5, 1) / m;
            eta(1) = -Muu(5, 0) / m;
            eta(2) = Muu(4, 0) / m;
        }
        for (int i = 0; i < rho.extent_int(0); ++i) {
            for (int j = 0; j < rho.extent_int(1); ++j) {
                rho(i, j) = Muu(i + 3, j + 3);
            }
        }
        VecTilde(eta, eta_tilde);
    }
};

}  // namespace openturbine
