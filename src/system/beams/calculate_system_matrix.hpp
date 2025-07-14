#pragma once

#include <KokkosBatched_Copy_Decl.hpp>
#include <KokkosBatched_Gemm_Decl.hpp>
#include <Kokkos_Core.hpp>

namespace openturbine::beams {

template <typename DeviceType>
struct CalculateSystemMatrix {
    size_t element;
    size_t num_nodes;
    typename Kokkos::View<double* [6][6], DeviceType>::const_type tangent;
    typename Kokkos::View<size_t**, DeviceType>::const_type node_state_indices;
    typename Kokkos::View<double** [6][6], DeviceType>::const_type stiffness_matrix_terms;
    typename Kokkos::View<double** [6][6], DeviceType>::const_type inertia_matrix_terms;
    Kokkos::View<double*** [6][6], DeviceType> system_matrix_terms;

    KOKKOS_FUNCTION
    void operator()(size_t node_12) const {
        const auto node_1 = node_12 / num_nodes;
        const auto node_2 = node_12 % num_nodes;
        const auto node_T = node_state_indices(element, node_2);

        auto S_data = Kokkos::Array<double, 36>{};
        auto T_data = Kokkos::Array<double, 36>{};
        auto STpI_data = Kokkos::Array<double, 36>{};

        const auto S = Kokkos::View<double[6][6], DeviceType>(S_data.data());
        const auto T = Kokkos::View<double[6][6], DeviceType>(T_data.data());
        const auto STpI = Kokkos::View<double[6][6], DeviceType>(STpI_data.data());

        KokkosBatched::SerialCopy<>::invoke(
            Kokkos::subview(stiffness_matrix_terms, node_1, node_2, Kokkos::ALL, Kokkos::ALL), S
        );
        KokkosBatched::SerialCopy<>::invoke(
            Kokkos::subview(tangent, node_T, Kokkos::ALL, Kokkos::ALL), T
        );
        KokkosBatched::SerialCopy<>::invoke(
            Kokkos::subview(inertia_matrix_terms, node_1, node_2, Kokkos::ALL, Kokkos::ALL), STpI
        );

        KokkosBatched::SerialGemm<
            KokkosBatched::Trans::NoTranspose, KokkosBatched::Trans::NoTranspose,
            KokkosBatched::Algo::Gemm::Default>::invoke(1., S, T, 1., STpI);

        KokkosBatched::SerialCopy<>::invoke(
            STpI,
            Kokkos::subview(system_matrix_terms, element, node_1, node_2, Kokkos::ALL, Kokkos::ALL)
        );
    }
};

}  // namespace openturbine::beams
