#pragma once

#include <KokkosBatched_Copy_Decl.hpp>
#include <KokkosBatched_Gemm_Decl.hpp>
#include <KokkosBlas.hpp>
#include <Kokkos_Core.hpp>

#include "math/vector_operations.hpp"

namespace openturbine {

template <typename DeviceType>
struct CalculateTangentOperator {
    double h;
    typename Kokkos::View<double* [6], DeviceType>::const_type q_delta;
    Kokkos::View<double* [6][6], DeviceType> T_gbl;

    KOKKOS_FUNCTION
    void operator()(const int node) const {
        auto T_data = Kokkos::Array<double, 36>{};
        const auto T = Kokkos::View<double[6][6], DeviceType>(T_data.data());

        auto rv_data = Kokkos::Array<double, 3>{};
        auto rv = Kokkos::View<double[3], DeviceType>{rv_data.data()};

        auto m1_data = Kokkos::Array<double, 9>{};
        auto m1 = Kokkos::View<double[3][3], DeviceType>(m1_data.data());
        auto m2_data = Kokkos::Array<double, 9>{};
        auto m2 = Kokkos::View<double[3][3], DeviceType>(m2_data.data());

        for (auto k = 0U; k < 6U; ++k) {
            T(k, k) = 1.;
        }

        KokkosBlas::serial_axpy(h, Kokkos::subview(q_delta, node, Kokkos::make_pair(3, 6)), rv);
        auto phi = KokkosBlas::serial_nrm2(rv);
        const auto tmp1 = (phi > 1.e-16) ? (Kokkos::cos(phi) - 1.) / (phi * phi) : 0.;
        const auto tmp2 = (phi > 1.e-16) ? (1. - Kokkos::sin(phi) / phi) / (phi * phi) : 0.;

        VecTilde(rv, m2);
        KokkosBatched::SerialCopy<>::invoke(m2, m1);
        KokkosBatched::SerialGemm<
            KokkosBatched::Trans::NoTranspose, KokkosBatched::Trans::NoTranspose,
            KokkosBatched::Algo::Gemm::Default>::invoke(tmp2, m2, m2, tmp1, m1);

        KokkosBlas::serial_axpy(
            1., m1, Kokkos::subview(T, Kokkos::make_pair(3, 6), Kokkos::make_pair(3, 6))
        );

        KokkosBatched::SerialCopy<KokkosBatched::Trans::Transpose>::invoke(
            T, Kokkos::subview(T_gbl, node, Kokkos::ALL, Kokkos::ALL)
        );
    }
};

}  // namespace openturbine
