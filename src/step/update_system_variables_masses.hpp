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
#include "system/masses/calculate_quadrature_point_values.hpp"
#include "system/masses/copy_to_quadrature_points.hpp"
#include "system/masses/update_node_state.hpp"

namespace openturbine {

inline void UpdateSystemVariablesMasses(
    const StepParameters& parameters, const Masses& masses, State& state
) {
    auto region = Kokkos::Profiling::ScopedRegion("Update System Variables Masses");

    // Update the node states for masses to get the current position/rotation
    Kokkos::parallel_for(
        "masses::UpdateNodeState", masses.num_elems,
        masses::UpdateNodeState{
            masses.state_indices, masses.node_u, masses.node_u_dot, masses.node_u_ddot, state.q,
            state.v, state.vd
        }
    );

    // Calculate some ancillary values (angular velocity - omega, angular acceleration - omega_dot,
    // linear acceleration - u_ddot etc.) that will be required by system kernels
    Kokkos::parallel_for(
        "masses::CopyToQuadraturePoints", masses.num_elems,
        masses::CopyToQuadraturePoints{
            masses.node_x0, masses.node_u, masses.node_u_dot, masses.node_u_ddot, masses.qp_x0,
            masses.qp_r0, masses.qp_u, masses.qp_r, masses.qp_u_ddot, masses.qp_omega,
            masses.qp_omega_dot
        }
    );

    Kokkos::parallel_for(
        "masses::CalculateQPPosition", masses.num_elems,
        masses::CalculateQPPosition{
            masses.qp_x0, masses.qp_u, masses.qp_r0, masses.qp_r, masses.qp_x
        }
    );

    // Calculate system variables by executing the system kernels and perform assembly
    Kokkos::parallel_for(
        "masses::CalculateQuadraturePointValues", masses.num_elems,
        masses::CalculateQuadraturePointValues{
            masses.gravity, masses.qp_Mstar, masses.qp_x, masses.qp_u_ddot, masses.qp_omega,
            masses.qp_omega_dot, masses.qp_eta, masses.qp_rho, masses.qp_eta_tilde,
            masses.qp_omega_tilde, masses.qp_omega_dot_tilde, masses.qp_Fi, masses.qp_Fg,
            masses.qp_RR0, masses.qp_Muu, masses.qp_Guu, masses.qp_Kuu
        }
    );

    AssembleResidualVectorMasses(masses);
    AssembleStiffnessMatrixMasses(masses);
    AssembleInertiaMatrixMasses(masses, parameters.beta_prime, parameters.gamma_prime);
}

}  // namespace openturbine
