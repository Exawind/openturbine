#pragma once

#include <Kokkos_Core.hpp>

#include "types.hpp"
#include "MatrixOperations.hpp"

namespace openturbine {

struct CalculateCuu {
    View_Nx6x6::const_type qp_RR0_;    //
    View_Nx6x6::const_type qp_Cstar_;  //
    View_Nx6x6 qp_Cuu_;                //
    View_Nx6x6 qp_Ctmp_;               //

    KOKKOS_FUNCTION
    void operator()(const int i_qp) const {
        auto RR0 = Kokkos::subview(qp_RR0_, i_qp, Kokkos::ALL, Kokkos::ALL);
        auto Cstar = Kokkos::subview(qp_Cstar_, i_qp, Kokkos::ALL, Kokkos::ALL);
        auto Cuu = Kokkos::subview(qp_Cuu_, i_qp, Kokkos::ALL, Kokkos::ALL);
        auto Ctmp = Kokkos::subview(qp_Ctmp_, i_qp, Kokkos::ALL, Kokkos::ALL);
        MatMulAB(RR0, Cstar, Ctmp);
        MatMulABT(Ctmp, RR0, Cuu);
    }
};

}
