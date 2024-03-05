#pragma once

#include <KokkosBlas.hpp>
#include <Kokkos_Core.hpp>

#include "src/gebt_poc/state.h"
#include "src/gebt_poc/types.hpp"
#include "src/gen_alpha_poc/utilities.h"

namespace openturbine::gebt_poc {

inline void NodalInertialForces(
    View1D_LieAlgebra::const_type velocity, View1D_LieAlgebra::const_type acceleration,
    const MassMatrix& sectional_mass_matrix, View1D_LieAlgebra inertial_forces_fc
) {
    // The inertial forces vector is defined as
    // {inertial_forces}_6x1 = {
    //     mass * { u_dot_dot + (omega_dot_tilde + omega_tilde * omega_tilde) * eta }_3x1,
    //     { mass * eta_tilde * u_dot_dot + [rho] * omega_dot + omega_tilde * [rho] * omega }_3x1
    // }
    // where,
    // mass - 1x1 = scalar mass of the beam element (from the sectional mass matrix)
    // u_dot_dot - 3x1 = translational acceleration of the center of mass of the beam element
    // omega - 3x1 = angular velocity of the beam element
    // omega_dot - 3x1 = angular acceleration of the beam element
    // omega_tilde - 3x3 = skew symmetric matrix of omega
    // eta - 3x1 = center of mass of the beam element
    // eta_tilde - 3x3 = skew symmetric matrix of eta
    // rho - 3x3 = moment of inertia matrix of the beam element (from the sectional mass matrix)

    Kokkos::deep_copy(inertial_forces_fc, 0.);

    // Calculate mass, {eta}, and [rho] from the sectional mass matrix
    auto mass = sectional_mass_matrix.GetMass();
    auto eta = sectional_mass_matrix.GetCenterOfMass();
    auto rho = sectional_mass_matrix.GetMomentOfInertia();

    // Calculate the first 3 elements of the inertial forces vector
    auto inertial_forces_fc_1 = Kokkos::subview(inertial_forces_fc, Kokkos::make_pair(0, 3));
    auto angular_velocity = Kokkos::subview(velocity, Kokkos::make_pair(3, 6));
    auto angular_velocity_tilde = gen_alpha_solver::create_cross_product_matrix(angular_velocity);
    auto accelaration = Kokkos::subview(acceleration, Kokkos::make_pair(0, 3));
    auto angular_acceleration = Kokkos::subview(acceleration, Kokkos::make_pair(3, 6));
    auto angular_acceleration_tilde =
        gen_alpha_solver::create_cross_product_matrix(angular_acceleration);

    auto temp = View2D_3x3("temp");
    KokkosBlas::gemm("N", "N", 1., angular_velocity_tilde, angular_velocity_tilde, 0., temp);
    KokkosBlas::axpy(1., angular_acceleration_tilde, temp);
    KokkosBlas::gemv("N", mass, temp, eta, 0., inertial_forces_fc_1);
    KokkosBlas::axpy(mass, accelaration, inertial_forces_fc_1);

    // Calculate the last 3 elements of the inertial forces vector
    auto inertial_forces_fc_2 = Kokkos::subview(inertial_forces_fc, Kokkos::make_pair(3, 6));
    auto center_of_mass_tilde = gen_alpha_solver::create_cross_product_matrix(eta);

    KokkosBlas::gemv("N", 1., rho, angular_acceleration, 0., inertial_forces_fc_2);
    auto temp2 = View2D_3x3("temp2");
    KokkosBlas::gemm("N", "N", 1., angular_velocity_tilde, rho, 0., temp2);
    KokkosBlas::gemv("N", 1., temp2, angular_velocity, 1., inertial_forces_fc_2);
    KokkosBlas::gemv("N", mass, center_of_mass_tilde, accelaration, 1., inertial_forces_fc_2);
}

}  // namespace openturbine::gebt_poc