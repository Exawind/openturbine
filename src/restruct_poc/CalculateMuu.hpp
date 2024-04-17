#pragma once

#include <Kokkos_Core.hpp>

#include "MatrixOperations.hpp"
#include "types.hpp"

namespace openturbine {

struct CalculateMuu {
    View_Nx6x6::const_type qp_RR0_;    //
    View_Nx6x6::const_type qp_Mstar_;  //
    View_Nx6x6 qp_Muu_;                //
    View_Nx6x6 qp_Mtmp_;               //

    KOKKOS_FUNCTION
    void operator()(const int i_qp) const {
        auto RR0 = Kokkos::subview(qp_RR0_, i_qp, Kokkos::ALL, Kokkos::ALL);
        auto Mstar = Kokkos::subview(qp_Mstar_, i_qp, Kokkos::ALL, Kokkos::ALL);
        auto Muu = Kokkos::subview(qp_Muu_, i_qp, Kokkos::ALL, Kokkos::ALL);
        auto Mtmp = Kokkos::subview(qp_Mtmp_, i_qp, Kokkos::ALL, Kokkos::ALL);
        MatMulAB(RR0, Mstar, Mtmp);
        MatMulABT(Mtmp, RR0, Muu);
    }
};

}  // namespace openturbine
