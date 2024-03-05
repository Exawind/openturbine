#pragma once

#include <KokkosBlas.hpp>
#include <Kokkos_Core.hpp>

#include "src/gebt_poc/types.hpp"
#include "src/gen_alpha_poc/utilities.h"

namespace openturbine::gebt_poc {

inline void NodalElasticForces(
    View1D_LieAlgebra::const_type sectional_strain, View2D_3x3::const_type rotation,
    View1D_LieGroup::const_type pos_vector_derivatives,
    View1D_LieGroup::const_type gen_coords_derivatives, View2D_6x6::const_type sectional_stiffness,
    View1D_LieAlgebra elastic_forces_fc, View1D_LieAlgebra elastic_forces_fd
) {
    // Calculate first part of the elastic forces i.e. F^C vector
    auto sectional_strain_next = View1D_LieAlgebra("sectional_strain_next");
    Kokkos::deep_copy(sectional_strain_next, sectional_strain);

    auto sectional_strain_next_1 = Kokkos::subview(sectional_strain_next, Kokkos::make_pair(0, 3));
    auto x0_prime = Kokkos::subview(pos_vector_derivatives, Kokkos::make_pair(0, 3));
    auto R_x0_prime = View1D_Vector("R_x0_prime");
    KokkosBlas::gemv("N", -1., rotation, x0_prime, 0., R_x0_prime);
    KokkosBlas::axpy(1., R_x0_prime, sectional_strain_next_1);

    Kokkos::deep_copy(elastic_forces_fc, 0.);
    KokkosBlas::gemv("N", 1., sectional_stiffness, sectional_strain_next, 0., elastic_forces_fc);

    // Calculate second part of the elastic forces i.e. F^D vector
    auto x0_prime_tilde = gen_alpha_solver::create_cross_product_matrix(x0_prime);
    auto u_prime_tilde = gen_alpha_solver::create_cross_product_matrix(
        Kokkos::subview(gen_coords_derivatives, Kokkos::make_pair(0, 3))
    );
    auto fd_values = View2D_3x3("fd_values");
    Kokkos::deep_copy(fd_values, x0_prime_tilde);
    KokkosBlas::axpy(1., u_prime_tilde, fd_values);

    Kokkos::deep_copy(elastic_forces_fd, 0.);
    auto elastic_force_fd_1 = Kokkos::subview(elastic_forces_fd, Kokkos::make_pair(3, 6));
    KokkosBlas::gemv(
        "T", 1., fd_values, Kokkos::subview(elastic_forces_fc, Kokkos::make_pair(0, 3)), 0.,
        elastic_force_fd_1
    );
}

}  // namespace openturbine::gebt_poc