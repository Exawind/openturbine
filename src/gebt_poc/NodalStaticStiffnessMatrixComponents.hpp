#pragma once

#include <Kokkos_Core.hpp>
#include <KokkosBlas.hpp>

#include "src/gen_alpha_poc/utilities.h"
#include "src/gebt_poc/types.hpp"

namespace openturbine::gebt_poc {

inline void CalculateOMatrix(
    View2D::const_type N_tilde, View2D::const_type M_tilde, View2D::const_type C11,
    View2D::const_type C21, View2D::const_type values, View2D O_matrix
) {
    // non_zero_terms_part_1 = -N_tilde + [C11] * (x_0_prime_tilde + u_prime_tilde)
    auto non_zero_terms_part_1 = View2D_3x3("non_zero_terms_part_1");
    KokkosBlas::gemm("N", "N", 1., C11, values, 0., non_zero_terms_part_1);
    KokkosBlas::axpy(-1., N_tilde, non_zero_terms_part_1);

    // non_zero_terms_part_2 = -M_tilde + [C21] * (x_0_prime_tilde  + u_prime_tilde)
    auto non_zero_terms_part_2 = View2D_3x3("non_zero_terms_part_2");
    KokkosBlas::gemm("N", "N", 1., C21, values, 0., non_zero_terms_part_2);
    KokkosBlas::axpy(-1., M_tilde, non_zero_terms_part_2);

    // Assemble the O matrix
    // [O]_6x6 = [
    //     [0]_3x3      -N_tilde + [C11] * (x_0_prime_tilde + u_prime_tilde)
    //     [0]_3x3      -M_tilde + [C21] * (x_0_prime_tilde  + u_prime_tilde)
    // ]
    Kokkos::deep_copy(O_matrix, 0.);
    auto O_matrix_1 = Kokkos::subview(O_matrix, Kokkos::make_pair(0, 3), Kokkos::make_pair(3, 6));
    Kokkos::deep_copy(O_matrix_1, non_zero_terms_part_1);
    auto O_matrix_2 = Kokkos::subview(O_matrix, Kokkos::make_pair(3, 6), Kokkos::make_pair(3, 6));
    Kokkos::deep_copy(O_matrix_2, non_zero_terms_part_2);
}

inline void CalculatePMatrix(
    View2D::const_type N_tilde, View2D::const_type C11, View2D::const_type C12,
    View2D::const_type values, View2D P_matrix
) {
    // non_zero_terms_part_3 = (x_0_prime_tilde + u_prime_tilde)^T * [C11]
    auto non_zero_terms_part_3 = View2D_3x3("non_zero_terms_part_3");
    KokkosBlas::gemm("T", "N", 1., values, C11, 0., non_zero_terms_part_3);
    KokkosBlas::axpy(1., N_tilde, non_zero_terms_part_3);

    // non_zero_terms_part_4 = (x_0_prime_tilde  + u_prime_tilde)^T * [C12]
    auto non_zero_terms_part_4 = View2D_3x3("non_zero_terms_part_4");
    KokkosBlas::gemm("T", "N", 1., values, C12, 0., non_zero_terms_part_4);

    // Assemble the P matrix
    // [P]_6x6 = [
    //                         [0]_3x3                                      [0]_3x3
    //     N_tilde + (x_0_prime_tilde + u_prime_tilde)^T * [C11]     (x_0_prime_tilde  +
    //     u_prime_tilde)^T * [C12]
    // ]
    Kokkos::deep_copy(P_matrix, 0.);
    auto P_matrix_1 = Kokkos::subview(P_matrix, Kokkos::make_pair(3, 6), Kokkos::make_pair(0, 3));
    Kokkos::deep_copy(P_matrix_1, non_zero_terms_part_3);
    auto P_matrix_2 = Kokkos::subview(P_matrix, Kokkos::make_pair(3, 6), Kokkos::make_pair(3, 6));
    Kokkos::deep_copy(P_matrix_2, non_zero_terms_part_4);
}

inline void CalculateQMatrix(
    View2D::const_type N_tilde, View2D::const_type C11, View2D::const_type values, View2D Q_matrix
) {
    // Assemble the Q matrix
    // [Q]_6x6 = [
    //     [0]_3x3                                          [0]_3x3
    //     [0]_3x3      (x_0_prime_tilde + u_prime_tilde)^T * (-N_tilde + [C11] * (x_0_prime_tilde  +
    //     u_prime_tilde))
    // ]
    auto val = View2D_3x3("val");
    KokkosBlas::gemm("N", "N", 1., C11, values, 0., val);
    KokkosBlas::axpy(-1., N_tilde, val);

    Kokkos::deep_copy(Q_matrix, 0.);
    auto non_zero_term = Kokkos::subview(Q_matrix, Kokkos::make_pair(3, 6), Kokkos::make_pair(3, 6));
    KokkosBlas::gemm("T", "N", 1., values, val, 0., non_zero_term);
}

inline void NodalStaticStiffnessMatrixComponents(
    View1D_LieAlgebra::const_type elastic_force_fc,
    View1D_LieGroup::const_type pos_vector_derivatives,
    View1D_LieGroup::const_type gen_coords_derivatives, View2D_6x6::const_type sectional_stiffness,
    View2D_6x6 O_matrix, View2D_6x6 P_matrix, View2D_6x6 Q_matrix
) {
    auto x0_prime_tilde = gen_alpha_solver::create_cross_product_matrix(
        Kokkos::subview(pos_vector_derivatives, Kokkos::make_pair(0, 3))
    );
    auto u_prime_tilde = gen_alpha_solver::create_cross_product_matrix(
        Kokkos::subview(gen_coords_derivatives, Kokkos::make_pair(0, 3))
    );

    auto values = View2D_3x3("values");
    Kokkos::deep_copy(values, x0_prime_tilde);
    KokkosBlas::axpy(1., u_prime_tilde, values);

    auto N = Kokkos::subview(elastic_force_fc, Kokkos::make_pair(0, 3));
    auto N_tilde = gen_alpha_solver::create_cross_product_matrix(N);
    auto M = Kokkos::subview(elastic_force_fc, Kokkos::make_pair(3, 6));
    auto M_tilde = gen_alpha_solver::create_cross_product_matrix(M);

    auto C11 =
        Kokkos::subview(sectional_stiffness, Kokkos::make_pair(0, 3), Kokkos::make_pair(0, 3));
    auto C12 =
        Kokkos::subview(sectional_stiffness, Kokkos::make_pair(0, 3), Kokkos::make_pair(3, 6));
    auto C21 =
        Kokkos::subview(sectional_stiffness, Kokkos::make_pair(3, 6), Kokkos::make_pair(0, 3));

    // Calculate the O, P, and Q matrices
    CalculateOMatrix(N_tilde, M_tilde, C11, C21, values, O_matrix);
    CalculatePMatrix(N_tilde, C11, C12, values, P_matrix);
    CalculateQMatrix(N_tilde, C11, values, Q_matrix);
}

}