#pragma once

#include <Kokkos_Core.hpp>

#include "assemble_inertia_matrix_masses.hpp"
#include "assemble_residual_vector_masses.hpp"
#include "assemble_stiffness_matrix_masses.hpp"
#include "elements/masses/masses.hpp"
#include "math/vector_operations.hpp"
#include "state/state.hpp"
#include "step_parameters.hpp"
#include "system/masses/calculate_QP_position.hpp"
#include "system/masses/calculate_RR0.hpp"
#include "system/masses/calculate_gravity_force.hpp"
#include "system/masses/calculate_gyroscopic_matrix.hpp"
#include "system/masses/calculate_inertia_stiffness_matrix.hpp"
#include "system/masses/calculate_inertial_force.hpp"
#include "system/masses/calculate_mass_matrix_components.hpp"
#include "system/masses/rotate_section_matrix.hpp"
#include "system/masses/update_node_state.hpp"

namespace openturbine {

inline void UpdateSystemVariablesMasses(
    const StepParameters& parameters, const Masses& masses, State& state
) {
    // Update the node states for masses to get the current position/rotation
    Kokkos::parallel_for(
        masses.num_elems,
        masses::UpdateNodeState{
            masses.state_indices, masses.node_u, masses.node_u_dot, masses.node_u_ddot, state.q,
            state.v, state.vd
        }
    );

    // Calculate some ancillary values (angular velocity - omega, angular acceleration - omega_dot,
    // linear acceleration - u_ddot etc.) that will be required by system kernels
    Kokkos::deep_copy(
        masses.qp_x0, Kokkos::subview(masses.node_x0, Kokkos::ALL, Kokkos::make_pair(0, 3))
    );
    Kokkos::deep_copy(
        masses.qp_u, Kokkos::subview(masses.node_u, Kokkos::ALL, Kokkos::make_pair(0, 3))
    );
    Kokkos::deep_copy(
        masses.qp_r0, Kokkos::subview(masses.node_x0, Kokkos::ALL, Kokkos::make_pair(3, 7))
    );
    Kokkos::deep_copy(
        masses.qp_u_ddot, Kokkos::subview(masses.node_u_ddot, Kokkos::ALL, Kokkos::make_pair(0, 3))
    );
    Kokkos::deep_copy(
        masses.qp_r, Kokkos::subview(masses.node_u, Kokkos::ALL, Kokkos::make_pair(3, 7))
    );
    Kokkos::deep_copy(
        masses.qp_omega, Kokkos::subview(masses.node_u_dot, Kokkos::ALL, Kokkos::make_pair(3, 6))
    );
    Kokkos::deep_copy(
        masses.qp_omega_dot,
        Kokkos::subview(masses.node_u_ddot, Kokkos::ALL, Kokkos::make_pair(3, 6))
    );

    Kokkos::parallel_for(
        masses.num_elems,
        KOKKOS_LAMBDA(size_t i_elem) {
            masses::CalculateQPPosition{i_elem,       masses.qp_x0, masses.qp_u,
                                        masses.qp_r0, masses.qp_r,  masses.qp_x}();
        }
    );

    // Calculate system variables by executing the system kernels and perform assembly
    Kokkos::parallel_for(
        masses.num_elems,
        KOKKOS_LAMBDA(const size_t i_elem) {
            // Calculate the global rotation matrix
            masses::CalculateRR0{i_elem, masses.qp_x, masses.qp_RR0}();

            // Transform mass matrix from material -> inertial frame
            masses::RotateSectionMatrix{i_elem, masses.qp_RR0, masses.qp_Mstar, masses.qp_Muu}();

            // Calculate mass matrix components i.e. eta, rho, and eta_tilde
            masses::CalculateMassMatrixComponents{
                i_elem, masses.qp_Muu, masses.qp_eta, masses.qp_rho, masses.qp_eta_tilde
            }();

            // Calculate gravity forces
            masses::CalculateGravityForce{
                i_elem, masses.gravity, masses.qp_Muu, masses.qp_eta_tilde, masses.qp_Fg
            }();

            // Calculate inertial forces i.e. forces due to linear + angular
            // acceleration
            masses::CalculateInertialForces{
                i_elem,
                masses.qp_Muu,
                masses.qp_u_ddot,
                masses.qp_omega,
                masses.qp_omega_dot,
                masses.qp_eta_tilde,
                masses.qp_omega_tilde,
                masses.qp_omega_dot_tilde,
                masses.qp_rho,
                masses.qp_eta,
                masses.qp_Fi
            }();

            // Calculate the gyroscopic/inertial damping matrix
            masses::CalculateGyroscopicMatrix{i_elem,          masses.qp_Muu,
                                              masses.qp_omega, masses.qp_omega_tilde,
                                              masses.qp_rho,   masses.qp_eta,
                                              masses.qp_Guu}();

            // Calculate the inertial stiffness matrix i.e. contributions from mass distribution and
            // rotational dynamics
            masses::CalculateInertiaStiffnessMatrix{
                i_elem,
                masses.qp_Muu,
                masses.qp_u_ddot,
                masses.qp_omega,
                masses.qp_omega_dot,
                masses.qp_omega_tilde,
                masses.qp_omega_dot_tilde,
                masses.qp_rho,
                masses.qp_eta,
                masses.qp_Kuu
            }();
        }
    );

    AssembleResidualVectorMasses(masses);
    AssembleStiffnessMatrixMasses(masses);
    AssembleInertiaMatrixMasses(masses, parameters.beta_prime, parameters.gamma_prime);
}

}  // namespace openturbine
