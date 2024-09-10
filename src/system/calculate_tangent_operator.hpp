#pragma once

#include <KokkosBatched_Gemm_Decl.hpp>
#include <KokkosBlas.hpp>
#include <Kokkos_Core.hpp>

#include "src/math/vector_operations.hpp"
#include "src/types.hpp"

namespace openturbine {

struct CalculateTangentOperator {
    double h;
    View_Nx6::const_type q_delta;
    View_Nx6x6 T;

    KOKKOS_FUNCTION
    void operator()(const int i_node) const {
        for (auto k = 0U; k < kLieAlgebraComponents; ++k) {
            for (auto n = 0U; n < kLieAlgebraComponents; ++n) {
                T(i_node, k, n) = 0.;
            }
        }

        for (auto k = 0U; k < kLieAlgebraComponents; ++k) {
            T(i_node, k, k) = 1.0;
        }

        auto rv_data = Kokkos::Array<double, 3>{};
        auto rv = Kokkos::View<double[3], Kokkos::MemoryTraits<Kokkos::Unmanaged>>{rv_data.data()};
        KokkosBlas::serial_axpy(h, Kokkos::subview(q_delta, i_node, Kokkos::make_pair(3, 6)), rv);
        auto phi = Kokkos::sqrt(rv(0) * rv(0) + rv(1) * rv(1) + rv(2) * rv(2));

        auto m1_data = Kokkos::Array<double, 9>{};
        auto m1 =
            Kokkos::View<double[3][3], Kokkos::MemoryTraits<Kokkos::Unmanaged>>(m1_data.data());
        auto m2_data = Kokkos::Array<double, 9>{};
        auto m2 =
            Kokkos::View<double[3][3], Kokkos::MemoryTraits<Kokkos::Unmanaged>>(m2_data.data());

        const auto tmp1 = (phi > 1.e-16) ? (Kokkos::cos(phi) - 1.) / (phi * phi) : 0.;
        const auto tmp2 = (phi > 1.e-16) ? (1. - Kokkos::sin(phi) / phi) / (phi * phi) : 0.;

        VecTilde(rv, m1);
        VecTilde(rv, m2);
        KokkosBatched::SerialGemm<
            KokkosBatched::Trans::NoTranspose, KokkosBatched::Trans::NoTranspose,
            KokkosBatched::Algo::Gemm::Default>::invoke(tmp2, m2, m2, tmp1, m1);

        for (auto k = 0U; k < 3U; ++k) {
            for (auto n = 0U; n < 3U; ++n) {
                T(i_node, k + 3, n + 3) += m1(k, n);
            }
        }
    }
};

}  // namespace openturbine
