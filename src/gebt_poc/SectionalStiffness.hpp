
#pragma once

#include <KokkosBlas.hpp>
#include <Kokkos_Core.hpp>

#include "src/gebt_poc/section.h"
#include "src/gebt_poc/types.hpp"

namespace openturbine::gebt_poc {

inline void SectionalStiffness(
    View2D_6x6::const_type stiffness, View2D_3x3::const_type rotation_0,
    View2D_3x3::const_type rotation, View2D_6x6 sectional_stiffness
) {
    auto total_rotation = View2D_3x3("total_rotation");
    KokkosBlas::gemm("N", "N", 1., rotation, rotation_0, 0., total_rotation);

    // rotation_matrix_6x6 = [
    //    [total_rotation]          [0]_3x3
    //        [0]_3x3           total_rotation
    // ]
    auto rotation_matrix = View2D_6x6("rotation_matrix");
    Kokkos::deep_copy(rotation_matrix, 0.);
    auto rotation_matrix_1 =
        Kokkos::subview(rotation_matrix, Kokkos::make_pair(0, 3), Kokkos::make_pair(0, 3));
    Kokkos::deep_copy(rotation_matrix_1, total_rotation);
    auto rotation_matrix_2 =
        Kokkos::subview(rotation_matrix, Kokkos::make_pair(3, 6), Kokkos::make_pair(3, 6));
    Kokkos::deep_copy(rotation_matrix_2, total_rotation);

    // Calculate the sectional stiffness matrix in inertial basis
    Kokkos::deep_copy(sectional_stiffness, 0.);
    auto stiffness_matrix_left_rot = View2D_6x6("temp");
    KokkosBlas::gemm(
        "N", "N", 1., rotation_matrix, stiffness, 0., stiffness_matrix_left_rot
    );
    KokkosBlas::gemm(
        "N", "T", 1., stiffness_matrix_left_rot, rotation_matrix, 0., sectional_stiffness
    );
}

}  // namespace openturbine::gebt_poc