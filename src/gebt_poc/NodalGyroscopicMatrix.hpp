#pragma once

#include <KokkosBlas.hpp>
#include <Kokkos_Core.hpp>

#include "src/gebt_poc/state.h"
#include "src/gebt_poc/types.hpp"
#include "src/gen_alpha_poc/utilities.h"

namespace openturbine::gebt_poc {

inline void NodalGyroscopicMatrix(
    View1D_LieAlgebra::const_type velocity, const MassMatrix& sectional_mass_matrix,
    View2D_6x6 gyroscopic_matrix
) {
    // The Gyroscopic matrix is defined as
    // {gyroscopic_matrix}_6x6 = [
    //     [0]_3x3      ~[omega_tilde * mass * eta]^T + omega_tilde * mass * eta_tilde^T
    //     [0]_3x3                  omega_tilde * rho - ~[rho * omega]
    // ]
    // where,
    // mass - 1x1 = scalar mass of the beam element (from the sectional mass matrix)
    // eta - 3x1 = center of mass of the beam element
    // omega - 3x1 = angular velocity of the beam element
    // omega_tilde - 3x3 = skew symmetric matrix of omega
    // eta_tilde - 3x3 = skew symmetric matrix of eta
    // rho - 3x3 = moment of inertia matrix of the beam element (from the sectional mass matrix)

    Kokkos::deep_copy(gyroscopic_matrix, 0.);

    // Calculate mass, {eta}, and [rho] from the sectional mass matrix
    auto mass = sectional_mass_matrix.GetMass();
    auto eta = sectional_mass_matrix.GetCenterOfMass();
    auto rho = sectional_mass_matrix.GetMomentOfInertia();

    // Calculate the top right block i.e. quadrant 1 of the gyroscopic matrix
    auto gyroscopic_matrix_q1 =
        Kokkos::subview(gyroscopic_matrix, Kokkos::make_pair(0, 3), Kokkos::make_pair(3, 6));
    auto angular_velocity = Kokkos::subview(velocity, Kokkos::make_pair(3, 6));
    auto angular_velocity_tilde = gen_alpha_solver::create_cross_product_matrix(angular_velocity);
    auto center_of_mass_tilde = gen_alpha_solver::create_cross_product_matrix(eta);

    auto temp1 = Kokkos::View<double[3]>("temp1");
    KokkosBlas::gemv("N", mass, angular_velocity_tilde, eta, 0., temp1);
    auto gyroscopic_matrix_q1_part1 =
        gen_alpha_solver::transpose_matrix(gen_alpha_solver::create_cross_product_matrix(temp1));
    KokkosBlas::gemm(
        "N", "T", mass, angular_velocity_tilde, center_of_mass_tilde, 0., gyroscopic_matrix_q1
    );
    KokkosBlas::axpy(1., gyroscopic_matrix_q1_part1, gyroscopic_matrix_q1);

    // Calculate the bottom right block i.e. quadrant 4 of the gyroscopic matrix
    auto gyroscopic_matrix_q4 =
        Kokkos::subview(gyroscopic_matrix, Kokkos::make_pair(3, 6), Kokkos::make_pair(3, 6));
    KokkosBlas::gemm("N", "N", 1., angular_velocity_tilde, rho, 0., gyroscopic_matrix_q4);
    auto temp2 = Kokkos::View<double[3]>("temp2");
    KokkosBlas::gemv("N", 1., rho, angular_velocity, 1., temp2);
    auto gyroscopic_matrix_q4_part2 = gen_alpha_solver::create_cross_product_matrix(temp2);
    KokkosBlas::axpy(-1., gyroscopic_matrix_q4_part2, gyroscopic_matrix_q4);
}

}  // namespace openturbine::gebt_poc