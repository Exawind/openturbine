#pragma once

#include <KokkosBlas.hpp>
#include <Kokkos_Core.hpp>

#include "src/gebt_poc/state.h"
#include "src/gebt_poc/types.hpp"
#include "src/gebt_poc/MassMatrix.hpp"
#include "src/gen_alpha_poc/utilities.h"

namespace openturbine::gebt_poc {
inline void NodalDynamicStiffnessMatrix(
    View1D_LieAlgebra::const_type velocity, View1D_LieAlgebra::const_type acceleration,
    View2D_6x6::const_type sectional_mass_matrix, View2D_6x6 stiffness_matrix
) {
    // The dynamic stiffness matrix is defined as
    // {dyn_stiffness_matrix}_6x6 = [
    //     [0]_3x3      (omega_dot_tilde + omega_tilde * omega_tilde) * mass * eta_tilde^T
    //
    //     [0]_3x3               acceleration_tilde * mass * eta_tilde + (rho * omega_dot_tilde  -
    //                      ~[rho * omega_dot]) + omega_tilde * (rho * omega_tilde - ~[rho * omega])
    // ]
    // where,
    // mass - 1x1 = scalar mass of the beam element (from the sectional mass matrix)
    // u_dot_dot - 3x1 = translational acceleration of the center of mass of the beam element
    // omega - 3x1 = angular velocity of the beam element
    // omega_dot - 3x1 = angular acceleration of the beam element
    // omega_tilde - 3x3 = skew symmetric matrix of omega
    // eta - 3x1 = center of mass of the beam element
    // eta_tilde - 3x3 = skew symmetric matrix of eta
    // rho - 3x3 = moment of inertia matrix of the beam element (from the sectional mass matrix)

    Kokkos::deep_copy(stiffness_matrix, 0.);

    // Calculate mass, {eta}, and [rho] from the sectional mass matrix
    auto mass = GetMass(sectional_mass_matrix);
    auto eta = GetCenterOfMass(sectional_mass_matrix);
    auto rho = GetMomentOfInertia(sectional_mass_matrix);

    // Calculate the top right block i.e. quadrant 1 of the dynamic stiffness matrix
    auto stiffness_matrix_q1 =
        Kokkos::subview(stiffness_matrix, Kokkos::make_pair(0, 3), Kokkos::make_pair(3, 6));
    auto angular_velocity = Kokkos::subview(velocity, Kokkos::make_pair(3, 6));
    auto angular_velocity_tilde = gen_alpha_solver::create_cross_product_matrix(angular_velocity);
    auto angular_acceleration = Kokkos::subview(acceleration, Kokkos::make_pair(3, 6));
    auto angular_acceleration_tilde =
        gen_alpha_solver::create_cross_product_matrix(angular_acceleration);
    auto center_of_mass_tilde = gen_alpha_solver::create_cross_product_matrix(eta);

    auto temp1 = View2D_3x3("temp1");
    KokkosBlas::gemm("N", "N", 1., angular_velocity_tilde, angular_velocity_tilde, 0., temp1);
    KokkosBlas::axpy(1., angular_acceleration_tilde, temp1);
    KokkosBlas::gemm("N", "T", mass, temp1, center_of_mass_tilde, 0., stiffness_matrix_q1);

    // Calculate the bottom right block i.e. quadrant 4 of the dynamic stiffness matrix
    auto stiffness_matrix_q4 =
        Kokkos::subview(stiffness_matrix, Kokkos::make_pair(3, 6), Kokkos::make_pair(3, 6));
    auto accelaration = Kokkos::subview(acceleration, Kokkos::make_pair(0, 3));
    auto accelaration_tilde = gen_alpha_solver::create_cross_product_matrix(accelaration);

    // part 1: acceleration_tilde * mass * eta_tilde
    auto temp2 = View2D_3x3("temp2");
    KokkosBlas::gemm("N", "N", mass, accelaration_tilde, center_of_mass_tilde, 0., temp2);
    // part 2: (rho * omega_dot_tilde  - ~[rho * omega_dot])
    auto temp3 = View2D_3x3("temp3");
    KokkosBlas::gemm("N", "N", 1., rho, angular_acceleration_tilde, 0., temp3);
    auto temp4 = View1D_Vector("temp4");
    KokkosBlas::gemv("N", 1., rho, angular_acceleration, 1., temp4);
    auto temp5 = gen_alpha_solver::create_cross_product_matrix(temp4);
    KokkosBlas::axpy(-1., temp5, temp3);
    // part 3: omega_tilde * (rho * omega_tilde - ~[rho * omega])
    auto temp6 = View2D_3x3("temp6");
    KokkosBlas::gemm("N", "N", 1., rho, angular_velocity_tilde, 0., temp6);
    auto temp7 = View1D_Vector("temp7");
    KokkosBlas::gemv("N", 1., rho, angular_velocity, 1., temp7);
    auto temp8 = gen_alpha_solver::create_cross_product_matrix(temp7);
    KokkosBlas::axpy(-1., temp8, temp6);
    KokkosBlas::gemm("N", "N", 1., angular_velocity_tilde, temp6, 0., stiffness_matrix_q4);

    KokkosBlas::axpy(1., temp2, stiffness_matrix_q4);
    KokkosBlas::axpy(1., temp3, stiffness_matrix_q4);
}
}  // namespace openturbine::gebt_poc