#pragma once

#include <Kokkos_Core.hpp>

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
#include "calculate_strain.hpp"
#include "calculate_temporary_variables.hpp"
#include "rotate_section_matrix.hpp"

namespace openturbine {
struct CalculateQuadraturePointValues {
    Kokkos::View<size_t*>::const_type num_qps_per_element;
    Kokkos::View<double[3]>::const_type gravity_;
    Kokkos::View<double** [3]>::const_type qp_u_prime_;
    Kokkos::View<double** [4]>::const_type qp_r_;
    Kokkos::View<double** [4]>::const_type qp_r_prime_;
    Kokkos::View<double** [4]>::const_type qp_r0_;
    Kokkos::View<double** [3]>::const_type qp_x0_prime_;
    Kokkos::View<double** [3]>::const_type qp_u_ddot_;
    Kokkos::View<double** [3]>::const_type qp_omega_;
    Kokkos::View<double** [3]>::const_type qp_omega_dot_;
    Kokkos::View<double** [6][6]>::const_type qp_Mstar_;
    Kokkos::View<double** [6][6]>::const_type qp_Cstar_;
    Kokkos::View<double** [6][6]> qp_RR0_;
    Kokkos::View<double** [6]> qp_strain_;
    Kokkos::View<double** [3]> qp_eta_;
    Kokkos::View<double** [3][3]> qp_rho_;
    Kokkos::View<double** [3][3]> qp_eta_tilde_;
    Kokkos::View<double** [3][3]> qp_x0pupSS_;
    Kokkos::View<double** [3][3]> qp_M_tilde_;
    Kokkos::View<double** [3][3]> qp_N_tilde_;
    Kokkos::View<double** [3][3]> qp_omega_tilde_;
    Kokkos::View<double** [3][3]> qp_omega_dot_tilde_;
    Kokkos::View<double** [6]> qp_FC_;
    Kokkos::View<double** [6]> qp_FD_;
    Kokkos::View<double** [6]> qp_FI_;
    Kokkos::View<double** [6]> qp_FG_;
    Kokkos::View<double** [6][6]> qp_Muu_;
    Kokkos::View<double** [6][6]> qp_Cuu_;
    Kokkos::View<double** [6][6]> qp_Ouu_;
    Kokkos::View<double** [6][6]> qp_Puu_;
    Kokkos::View<double** [6][6]> qp_Quu_;
    Kokkos::View<double** [6][6]> qp_Guu_;
    Kokkos::View<double** [6][6]> qp_Kuu_;

    KOKKOS_FUNCTION
    void operator()(Kokkos::TeamPolicy<>::member_type member) const {
        const auto i_elem = static_cast<size_t>(member.league_rank());
        const auto num_qps = num_qps_per_element(i_elem);

        Kokkos::parallel_for(
            Kokkos::TeamThreadRange(member, num_qps), CalculateRR0{i_elem, qp_r0_, qp_r_, qp_RR0_}
        );
        Kokkos::parallel_for(
            Kokkos::TeamThreadRange(member, num_qps),
            CalculateTemporaryVariables{i_elem, qp_x0_prime_, qp_u_prime_, qp_x0pupSS_}
        );
        member.team_barrier();
        Kokkos::parallel_for(
            Kokkos::TeamThreadRange(member, num_qps),
            RotateSectionMatrix{i_elem, qp_RR0_, qp_Mstar_, qp_Muu_}
        );

        Kokkos::parallel_for(
            Kokkos::TeamThreadRange(member, num_qps),
            RotateSectionMatrix{i_elem, qp_RR0_, qp_Cstar_, qp_Cuu_}
        );

        Kokkos::parallel_for(
            Kokkos::TeamThreadRange(member, num_qps),
            CalculateStrain{i_elem, qp_x0_prime_, qp_u_prime_, qp_r_, qp_r_prime_, qp_strain_}
        );
        member.team_barrier();
        Kokkos::parallel_for(
            Kokkos::TeamThreadRange(member, num_qps),
            CalculateMassMatrixComponents{i_elem, qp_Muu_, qp_eta_, qp_rho_, qp_eta_tilde_}
        );

        Kokkos::parallel_for(
            Kokkos::TeamThreadRange(member, num_qps),
            CalculateForceFC{i_elem, qp_Cuu_, qp_strain_, qp_FC_, qp_M_tilde_, qp_N_tilde_}
        );
        member.team_barrier();

        Kokkos::parallel_for(
            Kokkos::TeamThreadRange(member, num_qps),
            CalculateInertialForces{
                i_elem, qp_Muu_, qp_u_ddot_, qp_omega_, qp_omega_dot_, qp_eta_tilde_,
                qp_omega_tilde_, qp_omega_dot_tilde_, qp_rho_, qp_eta_, qp_FI_}
        );
        member.team_barrier();

        Kokkos::parallel_for(
            Kokkos::TeamThreadRange(member, num_qps),
            CalculateForceFD{i_elem, qp_x0pupSS_, qp_FC_, qp_FD_}
        );

        Kokkos::parallel_for(
            Kokkos::TeamThreadRange(member, num_qps),
            CalculateGravityForce{i_elem, gravity_, qp_Muu_, qp_eta_tilde_, qp_FG_}
        );

        Kokkos::parallel_for(
            Kokkos::TeamThreadRange(member, num_qps),
            CalculateOuu{i_elem, qp_Cuu_, qp_x0pupSS_, qp_M_tilde_, qp_N_tilde_, qp_Ouu_}
        );

        Kokkos::parallel_for(
            Kokkos::TeamThreadRange(member, num_qps),
            CalculatePuu{i_elem, qp_Cuu_, qp_x0pupSS_, qp_N_tilde_, qp_Puu_}
        );

        Kokkos::parallel_for(
            Kokkos::TeamThreadRange(member, num_qps),
            CalculateQuu{i_elem, qp_Cuu_, qp_x0pupSS_, qp_N_tilde_, qp_Quu_}
        );

        Kokkos::parallel_for(
            Kokkos::TeamThreadRange(member, num_qps),
            CalculateGyroscopicMatrix{
                i_elem, qp_Muu_, qp_omega_, qp_omega_tilde_, qp_rho_, qp_eta_, qp_Guu_}
        );

        Kokkos::parallel_for(
            Kokkos::TeamThreadRange(member, num_qps),
            CalculateInertiaStiffnessMatrix{
                i_elem, qp_Muu_, qp_u_ddot_, qp_omega_, qp_omega_dot_, qp_omega_tilde_,
                qp_omega_dot_tilde_, qp_rho_, qp_eta_, qp_Kuu_}
        );
    }
};
}  // namespace openturbine
