#pragma once

#include <KokkosBlas.hpp>
#include <Kokkos_Core.hpp>

#include <fstream>

#include "src/gebt_poc/types.hpp"
#include "src/gen_alpha_poc/quaternion.h"
#include "src/gen_alpha_poc/utilities.h"

namespace openturbine::gebt_poc {

inline void ElementalConstraintForcesGradientMatrix(
    View1D::const_type gen_coords, View1D::const_type position_vector,
    View2D constraints_gradient_matrix
) {
    auto translation_0 = Kokkos::subview(gen_coords, Kokkos::make_pair(0, 3));
    auto rotation_0 = Kokkos::subview(gen_coords, Kokkos::make_pair(3, 7));
    auto rotation_matrix_0 = gen_alpha_solver::EulerParameterToRotationMatrix(rotation_0);
    auto position_0 = Kokkos::subview(position_vector, Kokkos::make_pair(0, 3));

    // position_cross_prod_matrix = ~{position_0} + ~{translation_0}
    auto position_0_cross_prod_matrix = gen_alpha_solver::create_cross_product_matrix(position_0);
    auto translation_0_cross_prod_matrix =
        gen_alpha_solver::create_cross_product_matrix(translation_0);
    auto position_cross_prod_matrix = View2D_3x3("position_cross_prod_matrix");
    Kokkos::deep_copy(position_cross_prod_matrix, position_0_cross_prod_matrix);
    KokkosBlas::axpy(1., translation_0_cross_prod_matrix, position_cross_prod_matrix);

    // Assemble the constraint gradient matrix i.e. B matrix for the beam element
    // [B]_6x(n+1) = [
    //     [B11]_3x3              0            0   ....  0
    //     [B21]_3x3          [B22]_3x3        0   ....  0
    // ]
    // where
    // [B11]_3x3 = [1]_3x3
    // [B21]_3x3 = -[rotation_matrix_0]_3x3
    // [B22]_3x3 = -[rotation_matrix_0]_3x3 * [position_cross_prod_matrix]_3x3
    // n = order of the element

    auto B11 = Kokkos::subview(
        constraints_gradient_matrix, Kokkos::make_pair(0, 3), Kokkos::make_pair(0, 3)
    );
    auto B21 = Kokkos::subview(
        constraints_gradient_matrix, Kokkos::make_pair(3, 6), Kokkos::make_pair(0, 3)
    );
    auto B22 = Kokkos::subview(
        constraints_gradient_matrix, Kokkos::make_pair(3, 6), Kokkos::make_pair(3, 6)
    );

    Kokkos::parallel_for(
        1,
        KOKKOS_LAMBDA(std::size_t) {
            B11(0, 0) = 1.;
            B11(1, 1) = 1.;
            B11(2, 2) = 1.;
        }
    );

    KokkosBlas::scal(B21, -1., rotation_matrix_0);
    KokkosBlas::gemm("N", "N", -1., rotation_matrix_0, position_cross_prod_matrix, 0., B22);
}

inline void ConstraintGradientMatrixForRotatingBeam(
    View2D::const_type gen_coords, View1D applied_motion, View2D constraints_gradient_matrix
) {
    // Assemble the constraint gradient matrix i.e. B matrix
    // [B]_6x(n+1) = [
    //     [I]_3x3        [0]       [0]   ....  [0]
    //        [0]       [X]_3x3     [0]   ....  [0]
    // ]
    // where
    // [I]_3x3 = [1]_3x3
    // [0] = [0]_3x3
    // [X]_3x3 = (1/2) * (trace([R] * [R_BC]^T) * [I] - [R] * [R_BC]^T)
    // [R_BC] is the relative rotation matrix of the reference node
    // [R] is the relative rotation matrix of the constrained node

    // Relative rotation of reference node
    auto R_BC = openturbine::gen_alpha_solver::EulerParameterToRotationMatrix(
        Kokkos::subview(applied_motion, Kokkos::make_pair(3,7))
    );
    // Relative rotation of constrained node
    auto R = openturbine::gen_alpha_solver::EulerParameterToRotationMatrix(
        Kokkos::subview(gen_coords, 0, Kokkos::make_pair(3,7))
    );

    auto B11 = Kokkos::subview(
        constraints_gradient_matrix, Kokkos::make_pair(0, 3), Kokkos::make_pair(0, 3)
    );
    Kokkos::parallel_for(
        1,
        KOKKOS_LAMBDA(std::size_t) {
            B11(0, 0) = 1.;
            B11(1, 1) = 1.;
            B11(2, 2) = 1.;
        }
    );

    auto B22 = Kokkos::subview(
        constraints_gradient_matrix, Kokkos::make_pair(3, 6), Kokkos::make_pair(3, 6)
    );
    auto R_R_BC_T = View2D_3x3("R_R_BC_T");
    KokkosBlas::gemm("N", "T", 1., R, R_BC, 0., R_R_BC_T);
    Kokkos::parallel_for(
        1,
        KOKKOS_LAMBDA(std::size_t) {
            auto trace_R_R_BC_T = R_R_BC_T(0, 0) + R_R_BC_T(1, 1) + R_R_BC_T(2, 2);

            B22(0, 0) = 0.5 * (trace_R_R_BC_T - R_R_BC_T(0, 0));
            B22(0, 1) = -0.5 * R_R_BC_T(0, 1);
            B22(0, 2) = -0.5 * R_R_BC_T(0, 2);
            B22(1, 0) = -0.5 * R_R_BC_T(1, 0);
            B22(1, 1) = 0.5 * (trace_R_R_BC_T - R_R_BC_T(1, 1));
            B22(1, 2) = -0.5 * R_R_BC_T(1, 2);
            B22(2, 0) = -0.5 * R_R_BC_T(2, 0);
            B22(2, 1) = -0.5 * R_R_BC_T(2, 1);
            B22(2, 2) = 0.5 * (trace_R_R_BC_T - R_R_BC_T(2, 2));
        }
    );
}

}  // namespace openturbine::gebt_poc
