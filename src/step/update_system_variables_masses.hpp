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
    [[maybe_unused]] StepParameters& parameters, const Masses& masses, State& state
) {
    auto range_policy = Kokkos::TeamPolicy<>(static_cast<int>(masses.num_elems), Kokkos::AUTO());

    // Update the node states for masses to get the current position/rotation
    Kokkos::parallel_for(
        range_policy,
        UpdateNodeState{
            masses.num_nodes_per_element, masses.state_indices, masses.u, masses.u_dot,
            masses.u_ddot, state.q, state.v, state.a
        }
    );

    // Calculate some ancillary values (angular velocity - omega, angular acceleration - omega_dot,
    // linear acceleration - u_ddot) before calculating the mass element values
    auto x0 = Kokkos::View<double* [1][3]>("x0", masses.num_elems);
    Kokkos::deep_copy(x0, Kokkos::subview(masses.x0, Kokkos::ALL, Kokkos::ALL, Kokkos::make_pair(0, 3)));
    auto u = Kokkos::View<double* [1][3]>("u", masses.num_elems);
    Kokkos::deep_copy(u, Kokkos::subview(masses.u, Kokkos::ALL, Kokkos::ALL, Kokkos::make_pair(0, 3)));
    auto r0 = Kokkos::View<double* [1][4]>("r0", masses.num_elems);
    Kokkos::deep_copy(r0, Kokkos::subview(masses.x0, Kokkos::ALL, Kokkos::ALL, Kokkos::make_pair(3, 7)));
    auto r = Kokkos::View<double* [1][4]>("r", masses.num_elems);
    Kokkos::deep_copy(r, Kokkos::subview(masses.u, Kokkos::ALL, Kokkos::ALL, Kokkos::make_pair(3, 7)));

    auto omega = Kokkos::View<double* [1][3]>("omega", masses.num_elems);
    Kokkos::deep_copy(
        omega, Kokkos::subview(masses.u_dot, Kokkos::ALL, Kokkos::ALL, Kokkos::make_pair(3, 6))
    );
    auto omega_dot = Kokkos::View<double* [1][3]>("omega_dot", masses.num_elems);
    Kokkos::deep_copy(
        omega_dot, Kokkos::subview(masses.u_ddot, Kokkos::ALL, Kokkos::ALL, Kokkos::make_pair(3, 6))
    );
    auto u_ddot = Kokkos::View<double* [1][3]>("u_ddot", masses.num_elems);
    Kokkos::deep_copy(
        u_ddot, Kokkos::subview(masses.u_ddot, Kokkos::ALL, Kokkos::ALL, Kokkos::make_pair(0, 3))
    );

    Kokkos::parallel_for(masses.num_elems, KOKKOS_LAMBDA(size_t i_elem) {
        CalculateQPPosition{i_elem, x0, u, r0, r, masses.x}(0);
    });

    // Define some Views to store the skew-symmetric matrices
    auto eta_tilde = Kokkos::View<double* [1][3][3]>("eta_tilde", masses.num_elems);
    auto omega_tilde = Kokkos::View<double* [1][3][3]>("omega_tilde", masses.num_elems);
    auto omega_dot_tilde = Kokkos::View<double* [1][3][3]>("omega_dot_tilde", masses.num_elems);

    // Calculate system variables for mass elements
    Kokkos::parallel_for(
        range_policy,
        KOKKOS_LAMBDA(const Kokkos::TeamPolicy<>::member_type& member) {
            const auto i_elem = static_cast<size_t>(member.league_rank());

            // Calculate global rotation matrix
            Kokkos::parallel_for(1, CalculateRR0{i_elem, masses.x, masses.RR0});

            // Rotate mass matrix from material -> inertial frame
            Kokkos::parallel_for(
                1, RotateSectionMatrix{i_elem, masses.RR0, masses.Mstar, masses.Muu}
            );

            // Calculate mass matrix components
            Kokkos::parallel_for(
                1,
                CalculateMassMatrixComponents{i_elem, masses.Muu, masses.eta, masses.rho, eta_tilde}
            );

            // Calculate gravity forces
            Kokkos::parallel_for(
                1, CalculateGravityForce{i_elem, masses.gravity, masses.Muu, eta_tilde, masses.Fg}
            );

            // Calculate inertial forces
            Kokkos::parallel_for(
                1,
                CalculateInertialForces{
                    i_elem, masses.Muu, u_ddot, omega, omega_dot, eta_tilde, omega_tilde,
                    omega_dot_tilde, masses.rho, masses.eta, masses.Fi
                }
            );

            // Calculate gyroscopic/inertial damping matrix
            Kokkos::parallel_for(
                1,
                CalculateGyroscopicMatrix{
                    i_elem, masses.Muu, omega, omega_tilde, masses.rho, masses.eta, masses.Guu
                }
            );

            // Calculate inertia stiffness matrix
            Kokkos::parallel_for(
                1,
                CalculateInertiaStiffnessMatrix{
                    i_elem, masses.Muu, u_ddot, omega, omega_dot, omega_tilde, omega_dot_tilde,
                    masses.rho, masses.eta, masses.Kuu
                }
            );
        }
    );

    AssembleResidualVectorMasses(masses);
    AssembleStiffnessMatrixMasses(masses);
    AssembleInertiaMatrixMasses(masses, parameters.beta_prime, parameters.gamma_prime);
}

}  // namespace openturbine
