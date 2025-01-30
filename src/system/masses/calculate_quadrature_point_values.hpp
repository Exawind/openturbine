#pragma once

#include <Kokkos_Core.hpp>

#include "system/masses/calculate_RR0.hpp"
#include "system/masses/calculate_gravity_force.hpp"
#include "system/masses/calculate_gyroscopic_matrix.hpp"
#include "system/masses/calculate_inertia_stiffness_matrix.hpp"
#include "system/masses/calculate_inertial_force.hpp"
#include "system/masses/calculate_mass_matrix_components.hpp"
#include "system/masses/rotate_section_matrix.hpp"

namespace openturbine::masses {

struct CalculateQuadraturePointValues {
    Kokkos::View<double[3]>::const_type gravity;
    Kokkos::View<double* [6][6]>::const_type qp_Mstar;
    Kokkos::View<double* [7]>::const_type qp_x;
    Kokkos::View<double* [3]>::const_type qp_u_ddot;
    Kokkos::View<double* [3]>::const_type qp_omega;
    Kokkos::View<double* [3]>::const_type qp_omega_dot;

    Kokkos::View<double* [3]> qp_eta;
    Kokkos::View<double* [3][3]> qp_rho;
    Kokkos::View<double* [3][3]> qp_eta_tilde;
    Kokkos::View<double* [3][3]> qp_omega_tilde;
    Kokkos::View<double* [3][3]> qp_omega_dot_tilde;
    Kokkos::View<double* [6]> qp_Fi;
    Kokkos::View<double* [6]> qp_Fg;
    Kokkos::View<double* [6][6]> qp_RR0;
    Kokkos::View<double* [6][6]> qp_Muu;
    Kokkos::View<double* [6][6]> qp_Guu;
    Kokkos::View<double* [6][6]> qp_Kuu;

    KOKKOS_FUNCTION
    void operator()(size_t i_elem) const {
        // Calculate the global rotation matrix
        masses::CalculateRR0{i_elem, qp_x, qp_RR0}();

        // Transform mass matrix from material -> inertial frame
        masses::RotateSectionMatrix{i_elem, qp_RR0, qp_Mstar, qp_Muu}();

        // Calculate mass matrix components i.e. eta, rho, and eta_tilde
        masses::CalculateMassMatrixComponents{i_elem, qp_Muu, qp_eta, qp_rho, qp_eta_tilde}();

        // Calculate gravity forces
        masses::CalculateGravityForce{i_elem, gravity, qp_Muu, qp_eta_tilde, qp_Fg}();

        // Calculate inertial forces i.e. forces due to linear + angular
        // acceleration
        masses::CalculateInertialForces{
            i_elem,       qp_Muu,       qp_u_ddot,      qp_omega,
            qp_omega_dot, qp_eta_tilde, qp_omega_tilde, qp_omega_dot_tilde,
            qp_rho,       qp_eta,       qp_Fi
        }();

        // Calculate the gyroscopic/inertial damping matrix
        masses::CalculateGyroscopicMatrix{i_elem, qp_Muu, qp_omega, qp_omega_tilde,
                                          qp_rho, qp_eta, qp_Guu}();

        // Calculate the inertial stiffness matrix i.e. contributions from mass distribution and
        // rotational dynamics
        masses::CalculateInertiaStiffnessMatrix{
            i_elem, qp_Muu, qp_u_ddot, qp_omega, qp_omega_dot, qp_omega_tilde, qp_omega_dot_tilde,
            qp_rho, qp_eta, qp_Kuu
        }();
    }
};

}  // namespace openturbine::masses
