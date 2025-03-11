#pragma once

#include <KokkosBatched_Copy_Decl.hpp>
#include <KokkosBatched_Gemm_Decl.hpp>
#include <Kokkos_Core.hpp>

namespace openturbine::beams {

struct CalculateSystemMatrix {
    size_t i_elem;
    size_t num_nodes;
    Kokkos::View<double* [6][6]>::const_type tangent;
    Kokkos::View<size_t**>::const_type node_state_indices;
    Kokkos::View<double** [6][6]>::const_type stiffness_matrix_terms;
    Kokkos::View<double** [6][6]>::const_type inertia_matrix_terms;
    Kokkos::View<double*** [6][6]> system_matrix_terms;

    KOKKOS_FUNCTION
    void operator()(size_t ij_node) const {
        const auto i_node = ij_node / num_nodes;
        const auto j_node = ij_node % num_nodes;
        const auto T_node = node_state_indices(i_elem, j_node);

        auto S_data = Kokkos::Array<double, 36>{};
        auto T_data = Kokkos::Array<double, 36>{};
        auto STpI_data = Kokkos::Array<double, 36>{};

        const auto S = Kokkos::View<double[6][6]>(S_data.data());
        const auto T = Kokkos::View<double[6][6]>(T_data.data());
        const auto STpI = Kokkos::View<double[6][6]>(STpI_data.data());

        KokkosBatched::SerialCopy<>::invoke(
            Kokkos::subview(stiffness_matrix_terms, i_node, j_node, Kokkos::ALL, Kokkos::ALL), S
        );
        KokkosBatched::SerialCopy<>::invoke(
            Kokkos::subview(tangent, T_node, Kokkos::ALL, Kokkos::ALL), T
        );
        KokkosBatched::SerialCopy<>::invoke(
            Kokkos::subview(inertia_matrix_terms, i_node, j_node, Kokkos::ALL, Kokkos::ALL), STpI
        );

        KokkosBatched::SerialGemm<
            KokkosBatched::Trans::NoTranspose, KokkosBatched::Trans::NoTranspose,
            KokkosBatched::Algo::Gemm::Default>::invoke(1., S, T, 1., STpI);

        KokkosBatched::SerialCopy<>::invoke(
            STpI,
            Kokkos::subview(system_matrix_terms, i_elem, i_node, j_node, Kokkos::ALL, Kokkos::ALL)
        );
    }
};

}  // namespace openturbine::beams
