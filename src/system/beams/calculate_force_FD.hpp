#pragma once

#include <KokkosBlas.hpp>
#include <Kokkos_Core.hpp>

#include "math/vector_operations.hpp"

namespace openturbine {

struct CalculateForceFD {
    using Transpose = KokkosBlas::Trans::Transpose;
    using Default = KokkosBlas::Algo::Gemv::Default;
    using Gemv = KokkosBlas::SerialGemv<Transpose, Default>;
    size_t i_elem;
    Kokkos::View<double** [3][3]>::const_type x0pupSS_;
    Kokkos::View<double** [6]>::const_type qp_FC_;
    Kokkos::View<double** [6]> qp_FD_;

    KOKKOS_FUNCTION
    void operator()(int i_qp) const {
        auto x0pupSS = Kokkos::subview(x0pupSS_, i_elem, i_qp, Kokkos::ALL, Kokkos::ALL);
        auto FC = Kokkos::subview(qp_FC_, i_elem, i_qp, Kokkos::ALL);
        auto FD = Kokkos::subview(qp_FD_, i_elem, i_qp, Kokkos::ALL);

        auto N = Kokkos::subview(FC, Kokkos::make_pair(0, 3));
        for (int i = 0; i < FD.extent_int(0); ++i) {
            FD(i) = 0.;
        }
        Gemv::invoke(1., x0pupSS, N, 0., Kokkos::subview(FD, Kokkos::make_pair(3, 6)));
    }
};

}  // namespace openturbine
