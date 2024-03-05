#pragma once

#include <Kokkos_Core.hpp>
#include <KokkosBlas.hpp>

#include "src/gen_alpha_poc/quaternion.h"
#include "src/gebt_poc/types.hpp"

namespace openturbine::gebt_poc {

inline void NodalCurvature(
    View1D_LieGroup::const_type gen_coords, View1D_LieGroup::const_type gen_coords_derivative,
    View1D_Vector curvature
) {
    Kokkos::deep_copy(curvature, 0.);
    // curvature = B * q_prime
    auto b_matrix = Kokkos::View<double[3][4]>("b_matrix");
    gen_alpha_solver::BMatrixForQuaternions(
        b_matrix, Kokkos::subview(gen_coords, Kokkos::make_pair(3, 7))
    );
    auto q_prime = Kokkos::subview(gen_coords_derivative, Kokkos::make_pair(3, 7));
    KokkosBlas::gemv("N", 2., b_matrix, q_prime, 0., curvature);
}

}