#pragma once

#include <KokkosBlas.hpp>
#include <Kokkos_Core.hpp>

#include "src/restruct_poc/math/vector_operations.hpp"
#include "src/restruct_poc/types.hpp"

namespace openturbine {

struct CalculateForceFD {
    View_Nx3x3::const_type x0pupSS_;
    View_Nx6::const_type qp_FC_;
    View_Nx6 qp_FD_;

    KOKKOS_FUNCTION
    void operator()(int i_qp) const {
        auto x0pupSS = Kokkos::subview(x0pupSS_, i_qp, Kokkos::ALL, Kokkos::ALL);
        auto FC = Kokkos::subview(qp_FC_, i_qp, Kokkos::ALL);
        auto FD = Kokkos::subview(qp_FD_, i_qp, Kokkos::ALL);

        auto N = Kokkos::subview(FC, Kokkos::make_pair(0, 3));
        for (int i = 0; i < FD.extent_int(0); ++i) {
            FD(i) = 0.;
        }
        KokkosBlas::SerialGemv<KokkosBlas::Trans::Transpose, KokkosBlas::Algo::Gemv::Default>::
            invoke(1., x0pupSS, N, 0., Kokkos::subview(FD, Kokkos::make_pair(3, 6)));
    }
};

}  // namespace openturbine
