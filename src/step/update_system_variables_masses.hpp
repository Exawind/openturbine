#pragma once

#include <Kokkos_Core.hpp>

#include "assemble_inertia_matrix_masses.hpp"
#include "assemble_residual_vector_masses.hpp"
#include "assemble_stiffness_matrix_masses.hpp"
#include "step_parameters.hpp"

#include "src/elements/beams/calculate_QP_position.hpp"
#include "src/elements/masses/masses.hpp"
#include "src/math/vector_operations.hpp"
#include "src/state/state.hpp"
#include "src/system/calculate_RR0.hpp"
#include "src/system/calculate_inertial_force.hpp"
#include "src/system/update_node_state.hpp"

namespace openturbine {

inline void UpdateSystemVariablesMasses(
    const StepParameters& parameters, const Masses& masses, State& state
) {
    auto range_policy = Kokkos::TeamPolicy<>(static_cast<int>(masses.num_elems), Kokkos::AUTO());

    // Update the node states for masses to get the current position/rotation
    Kokkos::parallel_for(
        range_policy,
        UpdateNodeState{
            masses.num_nodes_per_element, masses.state_indices, masses.node_u, masses.node_u_dot,
            masses.node_u_ddot, state.q, state.v, state.vd
        }
    );

    // Calculate some ancillary values (angular velocity - omega, angular acceleration - omega_dot,
    // linear acceleration - u_ddot) before calculating the mass element values
    Kokkos::deep_copy(
        masses.qp_x0,
        Kokkos::subview(masses.node_x0, Kokkos::ALL, Kokkos::ALL, Kokkos::make_pair(0, 3))
    );
    Kokkos::deep_copy(
        masses.qp_u,
        Kokkos::subview(masses.node_u, Kokkos::ALL, Kokkos::ALL, Kokkos::make_pair(0, 3))
    );
    Kokkos::deep_copy(
        masses.qp_r0,
        Kokkos::subview(masses.node_x0, Kokkos::ALL, Kokkos::ALL, Kokkos::make_pair(3, 7))
    );
    Kokkos::deep_copy(
        masses.qp_u_ddot,
        Kokkos::subview(masses.node_u_ddot, Kokkos::ALL, Kokkos::ALL, Kokkos::make_pair(0, 3))
    );
    Kokkos::deep_copy(
        masses.qp_r,
        Kokkos::subview(masses.node_u, Kokkos::ALL, Kokkos::ALL, Kokkos::make_pair(3, 7))
    );
    Kokkos::deep_copy(
        masses.qp_omega,
        Kokkos::subview(masses.node_u_dot, Kokkos::ALL, Kokkos::ALL, Kokkos::make_pair(3, 6))
    );
    Kokkos::deep_copy(
        masses.qp_omega_dot,
        Kokkos::subview(masses.node_u_ddot, Kokkos::ALL, Kokkos::ALL, Kokkos::make_pair(3, 6))
    );

    Kokkos::parallel_for(
        masses.num_elems,
        KOKKOS_LAMBDA(size_t i_elem) {
            CalculateQPPosition{i_elem,       masses.qp_x0, masses.qp_u,
                                masses.qp_r0, masses.qp_r,  masses.qp_x}(0);
        }
    );

    // Calculate system variables for mass elements
    Kokkos::parallel_for(
        masses.num_elems,
        KOKKOS_LAMBDA(const size_t i_elem) {
            // Calculate global rotation matrix
            CalculateRR0{i_elem, masses.qp_x, masses.qp_RR0}(0);

            // Rotate mass matrix from material -> inertial frame
            RotateSectionMatrix{i_elem, masses.qp_RR0, masses.qp_Mstar, masses.qp_Muu}(0);

            // Calculate mass matrix components
            CalculateMassMatrixComponents{
                i_elem, masses.qp_Muu, masses.qp_eta, masses.qp_rho, masses.qp_eta_tilde
            }(0);

            // Calculate gravity forces
            CalculateGravityForce{
                i_elem, masses.gravity, masses.qp_Muu, masses.qp_eta_tilde, masses.qp_Fg
            }(0);

            // Calculate inertial forces
            CalculateInertialForces{
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
            }(0);

            // Calculate gyroscopic/inertial damping matrix
            CalculateGyroscopicMatrix{i_elem,          masses.qp_Muu,
                                      masses.qp_omega, masses.qp_omega_tilde,
                                      masses.qp_rho,   masses.qp_eta,
                                      masses.qp_Guu}(0);

            // Calculate inertia stiffness matrix
            CalculateInertiaStiffnessMatrix{
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
            }(0);
        }
    );

    AssembleResidualVectorMasses(masses);
    AssembleStiffnessMatrixMasses(masses);
    AssembleInertiaMatrixMasses(masses, parameters.beta_prime, parameters.gamma_prime);
}

}  // namespace openturbine
