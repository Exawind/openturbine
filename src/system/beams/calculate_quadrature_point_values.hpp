#pragma once

#include <Kokkos_Core.hpp>

#include "calculate_inertial_quadrature_point_values.hpp"
#include "calculate_stiffness_quadrature_point_values.hpp"
#include "calculate_system_matrix.hpp"
#include "integrate_inertia_matrix.hpp"
#include "integrate_residual_vector.hpp"
#include "integrate_stiffness_matrix.hpp"
#include "update_node_state.hpp"

namespace openturbine::beams {

template <typename DeviceType>
struct CalculateQuadraturePointValues {
    using member_type = typename Kokkos::TeamPolicy<typename DeviceType::execution_space>::member_type;
    double beta_prime_;
    double gamma_prime_;
    typename Kokkos::View<double* [7], DeviceType>::const_type Q;
    typename Kokkos::View<double* [6], DeviceType>::const_type V;
    typename Kokkos::View<double* [6], DeviceType>::const_type A;
    typename Kokkos::View<double* [6][6], DeviceType>::const_type tangent;
    typename Kokkos::View<size_t**, DeviceType>::const_type node_state_indices;
    typename Kokkos::View<size_t*, DeviceType>::const_type num_nodes_per_element;
    typename Kokkos::View<size_t*, DeviceType>::const_type num_qps_per_element;
    typename Kokkos::View<double**, DeviceType>::const_type qp_weight_;
    typename Kokkos::View<double**, DeviceType>::const_type qp_jacobian_;
    typename Kokkos::View<double***, DeviceType>::const_type shape_interp_;
    typename Kokkos::View<double***, DeviceType>::const_type shape_deriv_;
    typename Kokkos::View<double[3], DeviceType>::const_type gravity_;
    typename Kokkos::View<double** [6], DeviceType>::const_type node_FX_;
    typename Kokkos::View<double** [4], DeviceType>::const_type qp_r0_;
    typename Kokkos::View<double** [3], DeviceType>::const_type qp_x0_;
    typename Kokkos::View<double** [3], DeviceType>::const_type qp_x0_prime_;
    typename Kokkos::View<double** [6][6], DeviceType>::const_type qp_Mstar_;
    typename Kokkos::View<double** [6][6], DeviceType>::const_type qp_Cstar_;
    Kokkos::View<double** [6], DeviceType> qp_FE_;
    Kokkos::View<double** [6], DeviceType> residual_vector_terms_;
    Kokkos::View<double*** [6][6], DeviceType> system_matrix_terms_;

    KOKKOS_FUNCTION
    void operator()(member_type member) const {
        using simd_type = Kokkos::Experimental::simd<double>;
        const auto i_elem = static_cast<size_t>(member.league_rank());
        const auto num_nodes = num_nodes_per_element(i_elem);
        const auto num_qps = num_qps_per_element(i_elem);
        constexpr auto width = simd_type::size();
        const auto extra_component = num_nodes % width == 0U ? 0U : 1U;
        const auto simd_nodes = num_nodes / width + extra_component;

        const auto qp_range = Kokkos::TeamThreadRange(member, num_qps);
        const auto node_range = Kokkos::TeamThreadRange(member, num_nodes);
        const auto node_squared_range = Kokkos::TeamThreadRange(member, num_nodes * num_nodes);
        const auto node_squared_simd_range = Kokkos::TeamThreadRange(member, num_nodes * simd_nodes);

        const auto shape_interp =
            Kokkos::View<double**, Kokkos::LayoutLeft, DeviceType>(member.team_scratch(1), num_nodes, num_qps);
        const auto shape_deriv =
            Kokkos::View<double**, Kokkos::LayoutLeft, DeviceType>(member.team_scratch(1), num_nodes, num_qps);

        const auto qp_weight = Kokkos::View<double*, DeviceType>(member.team_scratch(1), num_qps);
        const auto qp_jacobian = Kokkos::View<double*, DeviceType>(member.team_scratch(1), num_qps);

        const auto node_u = Kokkos::View<double* [7], DeviceType>(member.team_scratch(1), num_nodes);
        const auto node_u_dot = Kokkos::View<double* [6], DeviceType>(member.team_scratch(1), num_nodes);
        const auto node_u_ddot = Kokkos::View<double* [6], DeviceType>(member.team_scratch(1), num_nodes);
        const auto node_FX = Kokkos::View<double* [6], DeviceType>(member.team_scratch(1), num_nodes);
        const auto qp_Fc = Kokkos::View<double* [6], DeviceType>(member.team_scratch(1), num_qps);
        const auto qp_Fd = Kokkos::View<double* [6], DeviceType>(member.team_scratch(1), num_qps);
        const auto qp_Fi = Kokkos::View<double* [6], DeviceType>(member.team_scratch(1), num_qps);
        const auto qp_Fe = Kokkos::View<double* [6], DeviceType>(member.team_scratch(1), num_qps);
        const auto qp_Fg = Kokkos::View<double* [6], DeviceType>(member.team_scratch(1), num_qps);

        const auto qp_Kuu = Kokkos::View<double* [6][6], DeviceType>(member.team_scratch(1), num_qps);
        const auto qp_Puu = Kokkos::View<double* [6][6], DeviceType>(member.team_scratch(1), num_qps);
        const auto qp_Cuu = Kokkos::View<double* [6][6], DeviceType>(member.team_scratch(1), num_qps);
        const auto qp_Ouu = Kokkos::View<double* [6][6], DeviceType>(member.team_scratch(1), num_qps);
        const auto qp_Quu = Kokkos::View<double* [6][6], DeviceType>(member.team_scratch(1), num_qps);
        const auto qp_Muu = Kokkos::View<double* [6][6], DeviceType>(member.team_scratch(1), num_qps);
        const auto qp_Guu = Kokkos::View<double* [6][6], DeviceType>(member.team_scratch(1), num_qps);

        const auto stiffness_matrix_terms =
            Kokkos::View<double** [6][6], DeviceType>(member.team_scratch(1), num_nodes, num_nodes);
        const auto inertia_matrix_terms =
            Kokkos::View<double** [6][6], DeviceType>(member.team_scratch(1), num_nodes, num_nodes);
        KokkosBatched::TeamVectorCopy<member_type>::invoke(
            member, Kokkos::subview(shape_interp_, i_elem, Kokkos::ALL, Kokkos::ALL), shape_interp
        );
        KokkosBatched::TeamVectorCopy<member_type>::invoke(
            member, Kokkos::subview(shape_deriv_, i_elem, Kokkos::ALL, Kokkos::ALL), shape_deriv
        );
        KokkosBatched::TeamVectorCopy<member_type>::invoke(
            member, Kokkos::subview(qp_FE_, i_elem, Kokkos::ALL, Kokkos::ALL), qp_Fe
        );
        KokkosBatched::TeamVectorCopy<member_type>::invoke(
            member, Kokkos::subview(node_FX_, i_elem, Kokkos::ALL, Kokkos::ALL), node_FX
        );

        KokkosBatched::TeamVectorCopy<
            member_type, KokkosBatched::Trans::NoTranspose,
            1>::invoke(member, Kokkos::subview(qp_weight_, i_elem, Kokkos::ALL), qp_weight);
        KokkosBatched::TeamVectorCopy<
            member_type, KokkosBatched::Trans::NoTranspose,
            1>::invoke(member, Kokkos::subview(qp_jacobian_, i_elem, Kokkos::ALL), qp_jacobian);

        const auto node_state_updater = beams::UpdateNodeStateElement<DeviceType>{
            i_elem, node_state_indices, node_u, node_u_dot, node_u_ddot, Q, V, A
        };
        Kokkos::parallel_for(node_range, node_state_updater);
        member.team_barrier();

        const auto inertia_quad_point_calculator = beams::CalculateInertialQuadraturePointValues<DeviceType>{
            i_elem,      shape_interp, gravity_, qp_r0_, qp_Mstar_, node_u, node_u_dot,
            node_u_ddot, qp_Fi,        qp_Fg,    qp_Muu, qp_Guu,    qp_Kuu
        };
        Kokkos::parallel_for(qp_range, inertia_quad_point_calculator);

        const auto stiffness_quad_point_calculator = beams::CalculateStiffnessQuadraturePointValues<DeviceType>{
            i_elem, qp_jacobian, shape_interp, shape_deriv, qp_r0_, qp_x0_prime_, qp_Cstar_,
            node_u, qp_Fc,       qp_Fd,        qp_Cuu,      qp_Ouu, qp_Puu,       qp_Quu
        };
        Kokkos::parallel_for(qp_range, stiffness_quad_point_calculator);
        member.team_barrier();

        const auto residual_integrator = beams::IntegrateResidualVectorElement<DeviceType>{
            i_elem, num_qps, qp_weight, qp_jacobian, shape_interp, shape_deriv,           node_FX,
            qp_Fc,  qp_Fd,   qp_Fi,     qp_Fe,       qp_Fg,        residual_vector_terms_
        };
        Kokkos::parallel_for(node_range, residual_integrator);

        const auto stiffness_matrix_integrator = beams::IntegrateStiffnessMatrixElement<DeviceType>{
            i_elem, num_nodes, num_qps, qp_weight, qp_jacobian, shape_interp,          shape_deriv,
            qp_Kuu, qp_Puu,    qp_Cuu,  qp_Ouu,    qp_Quu,      stiffness_matrix_terms
        };
        Kokkos::parallel_for(node_squared_simd_range, stiffness_matrix_integrator);

        const auto inertia_matrix_integrator = beams::IntegrateInertiaMatrixElement<DeviceType>{
            i_elem, num_nodes, num_qps,     qp_weight,    qp_jacobian,         shape_interp,
            qp_Muu, qp_Guu,    beta_prime_, gamma_prime_, inertia_matrix_terms
        };
        Kokkos::parallel_for(node_squared_simd_range, inertia_matrix_integrator);
        member.team_barrier();

        const auto system_matrix_calculator = beams::CalculateSystemMatrix<DeviceType>{
            i_elem,
            num_nodes,
            tangent,
            node_state_indices,
            stiffness_matrix_terms,
            inertia_matrix_terms,
            system_matrix_terms_
        };
        Kokkos::parallel_for(node_squared_range, system_matrix_calculator);
    }
};

}  // namespace openturbine::beams
