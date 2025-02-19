#pragma once

#include <Kokkos_Core.hpp>

#include "calculate_inertial_quadrature_point_values.hpp"
#include "calculate_stiffness_quadrature_point_values.hpp"
#include "integrate_inertia_matrix.hpp"
#include "integrate_residual_vector.hpp"
#include "integrate_stiffness_matrix.hpp"
#include "update_node_state.hpp"

namespace openturbine::beams {
struct CalculateQuadraturePointValues {
    double beta_prime_;
    double gamma_prime_;
    Kokkos::View<double* [7]>::const_type Q;
    Kokkos::View<double* [6]>::const_type V;
    Kokkos::View<double* [6]>::const_type A;
    Kokkos::View<size_t**>::const_type node_state_indices;
    Kokkos::View<size_t*>::const_type num_nodes_per_element;
    Kokkos::View<size_t*>::const_type num_qps_per_element;
    Kokkos::View<double**>::const_type qp_weight_;
    Kokkos::View<double**>::const_type qp_jacobian_;
    Kokkos::View<double***>::const_type shape_interp_;
    Kokkos::View<double***>::const_type shape_deriv_;
    Kokkos::View<double[3]>::const_type gravity_;
    Kokkos::View<double** [6]>::const_type node_FX_;
    Kokkos::View<double** [4]>::const_type qp_r0_;
    Kokkos::View<double** [3]>::const_type qp_x0_;
    Kokkos::View<double** [3]>::const_type qp_x0_prime_;
    Kokkos::View<double** [6][6]>::const_type qp_Mstar_;
    Kokkos::View<double** [6][6]>::const_type qp_Cstar_;
    Kokkos::View<double** [6]> qp_FE_;
    Kokkos::View<double** [6]> residual_vector_terms_;
    Kokkos::View<double*** [6][6]> stiffness_matrix_terms_;
    Kokkos::View<double*** [6][6]> inertia_matrix_terms_;

    KOKKOS_FUNCTION
    void operator()(Kokkos::TeamPolicy<>::member_type member) const {
        using simd_type = Kokkos::Experimental::native_simd<double>;
        const auto i_elem = static_cast<size_t>(member.league_rank());
        const auto num_nodes = num_nodes_per_element(i_elem);
        const auto num_qps = num_qps_per_element(i_elem);
        constexpr auto width = simd_type::size();
        const auto extra_component = num_nodes % width == 0U ? 0U : 1U;
        const auto simd_nodes = num_nodes / width + extra_component;

        const auto qp_range = Kokkos::TeamThreadRange(member, num_qps);
        const auto node_range = Kokkos::TeamThreadRange(member, num_nodes);
        const auto node_squared_range = Kokkos::TeamThreadRange(member, num_nodes * simd_nodes);

        const auto shape_interp =
            Kokkos::View<double**, Kokkos::LayoutLeft>(member.team_scratch(1), num_nodes, num_qps);
        const auto shape_deriv =
            Kokkos::View<double**, Kokkos::LayoutLeft>(member.team_scratch(1), num_nodes, num_qps);

        const auto qp_weight = Kokkos::View<double*>(member.team_scratch(1), num_qps);
        const auto qp_jacobian = Kokkos::View<double*>(member.team_scratch(1), num_qps);

        const auto node_u = Kokkos::View<double* [7]>(member.team_scratch(1), num_nodes);
        const auto node_u_dot = Kokkos::View<double* [6]>(member.team_scratch(1), num_nodes);
        const auto node_u_ddot = Kokkos::View<double* [6]>(member.team_scratch(1), num_nodes);
        const auto node_FX = Kokkos::View<double* [6]>(member.team_scratch(1), num_nodes);
        const auto qp_Fc = Kokkos::View<double* [6]>(member.team_scratch(1), num_qps);
        const auto qp_Fd = Kokkos::View<double* [6]>(member.team_scratch(1), num_qps);
        const auto qp_Fi = Kokkos::View<double* [6]>(member.team_scratch(1), num_qps);
        const auto qp_Fe = Kokkos::View<double* [6]>(member.team_scratch(1), num_qps);
        const auto qp_Fg = Kokkos::View<double* [6]>(member.team_scratch(1), num_qps);

        const auto qp_Kuu = Kokkos::View<double* [6][6]>(member.team_scratch(1), num_qps);
        const auto qp_Puu = Kokkos::View<double* [6][6]>(member.team_scratch(1), num_qps);
        const auto qp_Cuu = Kokkos::View<double* [6][6]>(member.team_scratch(1), num_qps);
        const auto qp_Ouu = Kokkos::View<double* [6][6]>(member.team_scratch(1), num_qps);
        const auto qp_Quu = Kokkos::View<double* [6][6]>(member.team_scratch(1), num_qps);
        const auto qp_Muu = Kokkos::View<double* [6][6]>(member.team_scratch(1), num_qps);
        const auto qp_Guu = Kokkos::View<double* [6][6]>(member.team_scratch(1), num_qps);

        KokkosBatched::TeamVectorCopy<Kokkos::TeamPolicy<>::member_type>::invoke(
            member, Kokkos::subview(shape_interp_, i_elem, Kokkos::ALL, Kokkos::ALL), shape_interp
        );
        KokkosBatched::TeamVectorCopy<Kokkos::TeamPolicy<>::member_type>::invoke(
            member, Kokkos::subview(shape_deriv_, i_elem, Kokkos::ALL, Kokkos::ALL), shape_deriv
        );
        KokkosBatched::TeamVectorCopy<Kokkos::TeamPolicy<>::member_type>::invoke(
            member, Kokkos::subview(qp_FE_, i_elem, Kokkos::ALL, Kokkos::ALL), qp_Fe
        );
        KokkosBatched::TeamVectorCopy<Kokkos::TeamPolicy<>::member_type>::invoke(
            member, Kokkos::subview(node_FX_, i_elem, Kokkos::ALL, Kokkos::ALL), node_FX
        );

        KokkosBatched::TeamVectorCopy<
            Kokkos::TeamPolicy<>::member_type, KokkosBatched::Trans::NoTranspose,
            1>::invoke(member, Kokkos::subview(qp_weight_, i_elem, Kokkos::ALL), qp_weight);
        KokkosBatched::TeamVectorCopy<
            Kokkos::TeamPolicy<>::member_type, KokkosBatched::Trans::NoTranspose,
            1>::invoke(member, Kokkos::subview(qp_jacobian_, i_elem, Kokkos::ALL), qp_jacobian);

        const auto node_state_updater = beams::UpdateNodeStateElement{
            i_elem, node_state_indices, node_u, node_u_dot, node_u_ddot, Q, V, A
        };
        Kokkos::parallel_for(node_range, node_state_updater);
        member.team_barrier();

        const auto inertia_quad_point_calculator = beams::CalculateInertialQuadraturePointValues{
            i_elem,      shape_interp, gravity_, qp_r0_, qp_Mstar_, node_u, node_u_dot,
            node_u_ddot, qp_Fi,        qp_Fg,    qp_Muu, qp_Guu,    qp_Kuu
        };
        Kokkos::parallel_for(qp_range, inertia_quad_point_calculator);

        const auto stiffness_quad_point_calculator = beams::CalculateStiffnessQuadraturePointValues{
            i_elem, qp_jacobian, shape_interp, shape_deriv, qp_r0_, qp_x0_prime_, qp_Cstar_,
            node_u, qp_Fc,       qp_Fd,        qp_Cuu,      qp_Ouu, qp_Puu,       qp_Quu
        };
        Kokkos::parallel_for(qp_range, stiffness_quad_point_calculator);
        member.team_barrier();

        const auto residual_integrator = IntegrateResidualVectorElement{
            i_elem, num_qps, qp_weight, qp_jacobian, shape_interp, shape_deriv,           node_FX,
            qp_Fc,  qp_Fd,   qp_Fi,     qp_Fe,       qp_Fg,        residual_vector_terms_
        };
        Kokkos::parallel_for(node_range, residual_integrator);

        const auto stiffness_matrix_integrator = IntegrateStiffnessMatrixElement{
            i_elem, num_nodes, num_qps, qp_weight, qp_jacobian, shape_interp,           shape_deriv,
            qp_Kuu, qp_Puu,    qp_Cuu,  qp_Ouu,    qp_Quu,      stiffness_matrix_terms_
        };
        Kokkos::parallel_for(node_squared_range, stiffness_matrix_integrator);

        const auto inertia_matrix_integrator = IntegrateInertiaMatrixElement{
            i_elem, num_nodes, num_qps,     qp_weight,    qp_jacobian,          shape_interp,
            qp_Muu, qp_Guu,    beta_prime_, gamma_prime_, inertia_matrix_terms_
        };
        Kokkos::parallel_for(node_squared_range, inertia_matrix_integrator);
    }
};

}  // namespace openturbine::beams
