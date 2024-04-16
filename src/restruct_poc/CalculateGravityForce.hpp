#pragma once

#include <Kokkos_Core.hpp>

#include "types.hpp"
#include "MatrixOperations.hpp"
#include "VectorOperations.hpp"

namespace openturbine {

struct CalculateGravityForce {
    View_3::const_type gravity;
    View_Nx6x6::const_type qp_Muu_;
    View_Nx3x3::const_type eta_tilde_;
    View_Nx3 v1_;
    View_Nx6 qp_FG_;

    KOKKOS_FUNCTION
    void operator()(int i_qp) const {
        auto Muu = Kokkos::subview(qp_Muu_, i_qp, Kokkos::ALL, Kokkos::ALL);
        auto eta_tilde = Kokkos::subview(eta_tilde_, i_qp, Kokkos::ALL, Kokkos::ALL);
        auto V1 = Kokkos::subview(v1_, i_qp, Kokkos::ALL);
        auto FG = Kokkos::subview(qp_FG_, i_qp, Kokkos::ALL);
        auto m = Muu(0, 0);
        VecScale(gravity, m, V1);
        for (int i = 0; i < 3; ++i) {
            FG(i) = V1(i);
        }
        MatVecMulAB(eta_tilde, V1, Kokkos::subview(FG, Kokkos::make_pair(3, 6)));
    }
};

}
