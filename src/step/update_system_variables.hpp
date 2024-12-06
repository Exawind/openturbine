#pragma once

#include <Kokkos_Core.hpp>
#include <Kokkos_Profiling_ScopedRegion.hpp>

#include "assemble_inertia_matrix.hpp"
#include "assemble_residual_vector.hpp"
#include "assemble_stiffness_matrix.hpp"
#include "step_parameters.hpp"

#include "src/elements/beams/interpolate_to_quadrature_points.hpp"
#include "src/elements/elements.hpp"
#include "src/state/state.hpp"
#include "src/system/calculate_quadrature_point_values.hpp"
#include "src/system/update_node_state.hpp"

namespace openturbine {

inline void UpdateBeamVariables(
    std::shared_ptr<Beams> beams, State& state, double beta_prime, double gamma_prime
) {
    auto range_policy_beams =
        Kokkos::TeamPolicy<>(static_cast<int>(beams->num_elems), Kokkos::AUTO());
    Kokkos::parallel_for(
        "UpdateNodeState_Beams", range_policy_beams,
        UpdateNodeState{
            beams->num_nodes_per_element,
            beams->node_state_indices,
            beams->node_u,
            beams->node_u_dot,
            beams->node_u_ddot,
            state.q,
            state.v,
            state.vd,
        }
    );

    Kokkos::parallel_for(
        "InterpolateToQuadraturePoints", range_policy_beams,
        InterpolateToQuadraturePoints{
            beams->num_nodes_per_element, beams->num_qps_per_element, beams->shape_interp,
            beams->shape_deriv, beams->qp_jacobian, beams->node_u, beams->node_u_dot,
            beams->node_u_ddot, beams->qp_x0, beams->qp_r0, beams->qp_u, beams->qp_u_prime,
            beams->qp_r, beams->qp_r_prime, beams->qp_u_dot, beams->qp_omega, beams->qp_u_ddot,
            beams->qp_omega_dot, beams->qp_x
        }
    );

    Kokkos::parallel_for(
        "CalculateQuadraturePointValues", range_policy_beams,
        CalculateQuadraturePointValues{
            beams->num_qps_per_element,
            beams->gravity,
            beams->qp_u,
            beams->qp_u_prime,
            beams->qp_r,
            beams->qp_r_prime,
            beams->qp_r0,
            beams->qp_x0,
            beams->qp_x0_prime,
            beams->qp_u_ddot,
            beams->qp_omega,
            beams->qp_omega_dot,
            beams->qp_Mstar,
            beams->qp_Cstar,
            beams->qp_x,
            beams->qp_RR0,
            beams->qp_strain,
            beams->qp_eta,
            beams->qp_rho,
            beams->qp_eta_tilde,
            beams->qp_x0pupss,
            beams->qp_M_tilde,
            beams->qp_N_tilde,
            beams->qp_omega_tilde,
            beams->qp_omega_dot_tilde,
            beams->qp_Fc,
            beams->qp_Fd,
            beams->qp_Fi,
            beams->qp_Fe,
            beams->qp_Fg,
            beams->qp_Muu,
            beams->qp_Cuu,
            beams->qp_Ouu,
            beams->qp_Puu,
            beams->qp_Quu,
            beams->qp_Guu,
            beams->qp_Kuu
        }
    );

    AssembleResidualVector(beams);
    AssembleStiffnessMatrix(beams);
    AssembleInertiaMatrix(beams, beta_prime, gamma_prime);
}

inline void UpdateSystemVariables(
    StepParameters& parameters, const Elements& elements, State& state
) {
    auto region = Kokkos::Profiling::ScopedRegion("Update System Variables");

    // Update Beams variables
    if (elements.beams) {
        UpdateBeamVariables(elements.beams, state, parameters.beta_prime, parameters.gamma_prime);
    }

    // TODO: Update Masses variables
}

}  // namespace openturbine
