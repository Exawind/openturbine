#pragma once

#include <Kokkos_Core.hpp>
#include <Kokkos_Profiling_ScopedRegion.hpp>

#include "calculate_Ouu.hpp"
#include "calculate_Puu.hpp"
#include "calculate_Quu.hpp"
#include "calculate_RR0.hpp"
#include "calculate_force_FC.hpp"
#include "calculate_force_FD.hpp"
#include "calculate_gravity_force.hpp"
#include "calculate_gyroscopic_matrix.hpp"
#include "calculate_inertia_stiffness_matrix.hpp"
#include "calculate_inertial_forces.hpp"
#include "calculate_mass_matrix_components.hpp"
#include "calculate_node_forces.hpp"
#include "calculate_strain.hpp"
#include "calculate_temporary_variables.hpp"
#include "rotate_section_matrix.hpp"
#include "update_node_state.hpp"

#include "src/restruct_poc/beams/beams.hpp"
#include "src/restruct_poc/beams/interpolate_QP_acceleration.hpp"
#include "src/restruct_poc/beams/interpolate_QP_state.hpp"
#include "src/restruct_poc/beams/interpolate_QP_velocity.hpp"
#include "src/restruct_poc/types.hpp"

namespace openturbine {

inline void UpdateState(Beams& beams, View_Nx7 Q, View_Nx6 V, View_Nx6 A) {
    auto region = Kokkos::Profiling::ScopedRegion("Update State");
    Kokkos::parallel_for(
        "UpdateNodeState", Kokkos::MDRangePolicy{{0, 0}, {beams.num_nodes, kLieGroupComponents}},
        UpdateNodeState{
            beams.node_state_indices,
            beams.node_u,
            beams.node_u_dot,
            beams.node_u_ddot,
            Q,
            V,
            A,
        }
    );

    Kokkos::parallel_for(
        "InterpolateQpU", Kokkos::MDRangePolicy{{0, 0}, {beams.num_elems, beams.max_elem_qps}},
        InterpolateQPU{beams.elem_indices, beams.shape_interp, beams.node_u, beams.qp_u}
    );
    Kokkos::parallel_for(
        "InterpolateQpU_Prime", Kokkos::MDRangePolicy{{0, 0}, {beams.num_elems, beams.max_elem_qps}},
        InterpolateQPU_Prime{
            beams.elem_indices, beams.shape_deriv, beams.qp_jacobian, beams.node_u, beams.qp_u_prime}
    );
    Kokkos::parallel_for(
        "InterpolateQpR", Kokkos::MDRangePolicy{{0, 0}, {beams.num_elems, beams.max_elem_qps}},
        InterpolateQPR{beams.elem_indices, beams.shape_interp, beams.node_u, beams.qp_r}
    );
    Kokkos::parallel_for(
        "InterpolateQpR_Prime", Kokkos::MDRangePolicy{{0, 0}, {beams.num_elems, beams.max_elem_qps}},
        InterpolateQPR_Prime{
            beams.elem_indices, beams.shape_deriv, beams.qp_jacobian, beams.node_u, beams.qp_r_prime}
    );
    Kokkos::parallel_for(
        "InterpolateQPVelocity_Translation",
        Kokkos::MDRangePolicy{{0, 0}, {beams.num_elems, beams.max_elem_qps}},
        InterpolateQPVelocity_Translation{
            beams.elem_indices, beams.shape_interp, beams.node_u_dot, beams.qp_u_dot}
    );
    Kokkos::parallel_for(
        "InterpolateQPVelocity_Angular",
        Kokkos::MDRangePolicy{{0, 0}, {beams.num_elems, beams.max_elem_qps}},
        InterpolateQPVelocity_Angular{
            beams.elem_indices,
            beams.shape_interp,
            beams.node_u_dot,
            beams.qp_omega,
        }
    );
    Kokkos::parallel_for(
        "InterpolateQPAcceleration_Translation",
        Kokkos::MDRangePolicy{{0, 0}, {beams.num_elems, beams.max_elem_qps}},
        InterpolateQPAcceleration_Translation{
            beams.elem_indices, beams.shape_interp, beams.node_u_ddot, beams.qp_u_ddot}
    );
    Kokkos::parallel_for(
        "InterpolateQPAcceleration_Angular",
        Kokkos::MDRangePolicy{{0, 0}, {beams.num_elems, beams.max_elem_qps}},
        InterpolateQPAcceleration_Angular{
            beams.elem_indices,
            beams.shape_interp,
            beams.node_u_ddot,
            beams.qp_omega_dot,
        }
    );

    Kokkos::parallel_for(
        "CalculateRR0", beams.num_qps,
        CalculateRR0{
            beams.qp_r0,
            beams.qp_r,
            beams.qp_RR0,
        }
    );

    Kokkos::parallel_for(
        "RotateSectionMatrix", beams.num_qps,
        RotateSectionMatrix{beams.qp_RR0, beams.qp_Mstar, beams.qp_Muu}
    );

    Kokkos::parallel_for(
        "RotateSectionMatrix", beams.num_qps,
        RotateSectionMatrix{beams.qp_RR0, beams.qp_Cstar, beams.qp_Cuu}
    );

    Kokkos::parallel_for(
        "CalculateStrain", beams.num_qps,
        CalculateStrain{
            beams.qp_x0_prime,
            beams.qp_u_prime,
            beams.qp_r,
            beams.qp_r_prime,
            beams.qp_strain,
        }
    );

    Kokkos::parallel_for(
        "CalculateMassMatrixComponents", beams.num_qps,
        CalculateMassMatrixComponents{beams.qp_Muu, beams.qp_eta, beams.qp_rho, beams.qp_eta_tilde}
    );
    Kokkos::parallel_for(
        "CalculateTemporaryVariables", beams.num_qps,
        CalculateTemporaryVariables{beams.qp_x0_prime, beams.qp_u_prime, beams.qp_x0pupss}
    );
    Kokkos::parallel_for(
        "CalculateForceFC", beams.num_qps,
        CalculateForceFC{
            beams.qp_Cuu, beams.qp_strain, beams.qp_Fc, beams.qp_M_tilde, beams.qp_N_tilde}
    );
    Kokkos::parallel_for(
        "CalculateForceFD", beams.num_qps,
        CalculateForceFD{beams.qp_x0pupss, beams.qp_Fc, beams.qp_Fd}
    );
    Kokkos::parallel_for(
        "CalculateInertialForces", beams.num_qps,
        CalculateInertialForces{
            beams.qp_Muu, beams.qp_u_ddot, beams.qp_omega, beams.qp_omega_dot, beams.qp_eta_tilde,
            beams.qp_omega_tilde, beams.qp_omega_dot_tilde, beams.qp_rho, beams.qp_eta, beams.qp_Fi}
    );
    Kokkos::parallel_for(
        "CalculateGravityForce", beams.num_qps,
        CalculateGravityForce{beams.gravity, beams.qp_Muu, beams.qp_eta_tilde, beams.qp_Fg}
    );
    Kokkos::parallel_for(
        "CalculateOuu", beams.num_qps,
        CalculateOuu{
            beams.qp_Cuu, beams.qp_x0pupss, beams.qp_M_tilde, beams.qp_N_tilde, beams.qp_Ouu}
    );
    Kokkos::parallel_for(
        "CalculatePuu", beams.num_qps,
        CalculatePuu{beams.qp_Cuu, beams.qp_x0pupss, beams.qp_N_tilde, beams.qp_Puu}
    );
    Kokkos::parallel_for(
        "CalculateQuu", beams.num_qps,
        CalculateQuu{beams.qp_Cuu, beams.qp_x0pupss, beams.qp_N_tilde, beams.qp_Quu}
    );
    Kokkos::parallel_for(
        "CalculateGyroscopicMatrix", beams.num_qps,
        CalculateGyroscopicMatrix{
            beams.qp_Muu, beams.qp_omega, beams.qp_omega_tilde, beams.qp_rho, beams.qp_eta,
            beams.qp_Guu}
    );
    Kokkos::parallel_for(
        "CalculateInertiaStiffnessMatrix", beams.num_qps,
        CalculateInertiaStiffnessMatrix{
            beams.qp_Muu, beams.qp_u_ddot, beams.qp_omega, beams.qp_omega_dot, beams.qp_omega_tilde,
            beams.qp_omega_dot_tilde, beams.qp_rho, beams.qp_eta, beams.qp_Kuu}
    );

    Kokkos::parallel_for(
        "CalculateNodeForces",
        Kokkos::MDRangePolicy{{0, 0, 0}, {beams.num_elems, beams.max_elem_nodes, 6}},
        CalculateNodeForces{
            beams.elem_indices, beams.qp_weight, beams.qp_jacobian, beams.shape_interp,
            beams.shape_deriv, beams.qp_Fc, beams.qp_Fd, beams.qp_Fi, beams.qp_Fg, beams.node_FE,
            beams.node_FI, beams.node_FG}
    );
}

}  // namespace openturbine
