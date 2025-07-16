#pragma once

#include "beams.hpp"
#include "beams_input.hpp"
#include "calculate_jacobian.hpp"
#include "interpolate_QP_position.hpp"
#include "interpolate_QP_rotation.hpp"
#include "interpolate_to_quadrature_points.hpp"
#include "populate_element_views.hpp"

namespace openturbine {

template <typename DeviceType>
inline Beams<DeviceType> CreateBeams(const BeamsInput& beams_input, const std::vector<Node>& nodes) {
    Beams<DeviceType> beams(
        beams_input.NumElements(), beams_input.MaxElemNodes(), beams_input.MaxElemQuadraturePoints()
    );

    auto host_gravity = Kokkos::create_mirror_view(Kokkos::WithoutInitializing, beams.gravity);

    auto host_num_nodes_per_element =
        Kokkos::create_mirror_view(Kokkos::WithoutInitializing, beams.num_nodes_per_element);
    auto host_num_qps_per_element =
        Kokkos::create_mirror_view(Kokkos::WithoutInitializing, beams.num_qps_per_element);
    auto host_node_state_indices =
        Kokkos::create_mirror_view(Kokkos::WithoutInitializing, beams.node_state_indices);
    auto host_node_x0 = Kokkos::create_mirror_view(Kokkos::WithoutInitializing, beams.node_x0);
    auto host_node_u = Kokkos::create_mirror_view(Kokkos::WithoutInitializing, beams.node_u);
    auto host_node_u_dot = Kokkos::create_mirror_view(Kokkos::WithoutInitializing, beams.node_u_dot);
    auto host_node_u_ddot =
        Kokkos::create_mirror_view(Kokkos::WithoutInitializing, beams.node_u_ddot);

    auto host_qp_weight = Kokkos::create_mirror_view(Kokkos::WithoutInitializing, beams.qp_weight);
    auto host_qp_Mstar = Kokkos::create_mirror_view(Kokkos::WithoutInitializing, beams.qp_Mstar);
    auto host_qp_Cstar = Kokkos::create_mirror_view(Kokkos::WithoutInitializing, beams.qp_Cstar);

    auto host_shape_interp =
        Kokkos::create_mirror_view(Kokkos::WithoutInitializing, beams.shape_interp);
    auto host_shape_deriv =
        Kokkos::create_mirror_view(Kokkos::WithoutInitializing, beams.shape_deriv);

    host_gravity(0) = beams_input.gravity[0];
    host_gravity(1) = beams_input.gravity[1];
    host_gravity(2) = beams_input.gravity[2];

    for (auto element = 0U; element < beams_input.NumElements(); ++element) {
        // Get number of nodes and quadrature points in element
        const auto num_nodes = beams_input.elements[element].node_ids.size();
        const auto num_qps = beams_input.elements[element].quadrature.size();

        // Create element indices and set in host mirror
        host_num_nodes_per_element(element) = num_nodes;
        host_num_qps_per_element(element) = num_qps;

        // Populate beam node->state indices
        for (auto node = 0U; node < num_nodes; ++node) {
            host_node_state_indices(element, node) = beams_input.elements[element].node_ids[node];
        }

        // Populate views for this element
        PopulateNodeX0(
            beams_input.elements[element], nodes,
            Kokkos::subview(host_node_x0, element, Kokkos::make_pair(0UL, num_nodes), Kokkos::ALL)
        );
        PopulateQPWeight(
            beams_input.elements[element],
            Kokkos::subview(host_qp_weight, element, Kokkos::make_pair(0UL, num_qps))
        );
        PopulateShapeFunctionValues(
            beams_input.elements[element], nodes,
            Kokkos::subview(
                host_shape_interp, element, Kokkos::make_pair(0UL, num_nodes),
                Kokkos::make_pair(0UL, num_qps)
            )
        );
        PopulateShapeFunctionDerivatives(
            beams_input.elements[element], nodes,
            Kokkos::subview(
                host_shape_deriv, element, Kokkos::make_pair(0UL, num_nodes),
                Kokkos::make_pair(0UL, num_qps)
            )
        );
        PopulateQPMstar(
            beams_input.elements[element],
            Kokkos::subview(
                host_qp_Mstar, element, Kokkos::make_pair(0UL, num_qps), Kokkos::ALL, Kokkos::ALL
            )
        );
        PopulateQPCstar(
            beams_input.elements[element],
            Kokkos::subview(
                host_qp_Cstar, element, Kokkos::make_pair(0UL, num_qps), Kokkos::ALL, Kokkos::ALL
            )
        );
    }

    Kokkos::deep_copy(beams.gravity, host_gravity);
    Kokkos::deep_copy(beams.num_nodes_per_element, host_num_nodes_per_element);
    Kokkos::deep_copy(beams.num_qps_per_element, host_num_qps_per_element);
    Kokkos::deep_copy(beams.node_state_indices, host_node_state_indices);
    Kokkos::deep_copy(beams.node_x0, host_node_x0);
    Kokkos::deep_copy(beams.node_u, host_node_u);
    Kokkos::deep_copy(beams.node_u_dot, host_node_u_dot);
    Kokkos::deep_copy(beams.node_u_ddot, host_node_u_ddot);
    Kokkos::deep_copy(beams.qp_weight, host_qp_weight);
    Kokkos::deep_copy(beams.qp_Mstar, host_qp_Mstar);
    Kokkos::deep_copy(beams.qp_Cstar, host_qp_Cstar);
    Kokkos::deep_copy(beams.shape_interp, host_shape_interp);
    Kokkos::deep_copy(beams.shape_deriv, host_shape_deriv);

    Kokkos::deep_copy(beams.node_FX, 0.);
    Kokkos::deep_copy(beams.qp_Fe, 0.);

    auto range_policy =
        Kokkos::RangePolicy<typename DeviceType::execution_space>(0, beams.num_elems);

    Kokkos::parallel_for(
        "InterpolateQPPosition", range_policy,
        InterpolateQPPosition<DeviceType>{
            beams.num_nodes_per_element, beams.num_qps_per_element, beams.shape_interp,
            beams.node_x0, beams.qp_x0
        }
    );

    Kokkos::parallel_for(
        "InterpolateQPRotation", range_policy,
        InterpolateQPRotation<DeviceType>{
            beams.num_nodes_per_element, beams.num_qps_per_element, beams.shape_interp,
            beams.node_x0, beams.qp_r0
        }
    );

    Kokkos::parallel_for(
        "CalculateJacobian", range_policy,
        CalculateJacobian<DeviceType>{
            beams.num_nodes_per_element,
            beams.num_qps_per_element,
            beams.shape_deriv,
            beams.node_x0,
            beams.qp_x0_prime,
            beams.qp_jacobian,
        }
    );

    auto team_policy = Kokkos::TeamPolicy<typename DeviceType::execution_space>(
        static_cast<int>(beams.num_elems), Kokkos::AUTO()
    );
    Kokkos::parallel_for(
        "InterpolateToQuadraturePoints", team_policy,
        InterpolateToQuadraturePoints<DeviceType>{
            beams.num_nodes_per_element, beams.num_qps_per_element, beams.shape_interp,
            beams.shape_deriv, beams.qp_jacobian, beams.node_u, beams.node_u_dot, beams.node_u_ddot,
            beams.qp_x0, beams.qp_r0, beams.qp_u, beams.qp_u_prime, beams.qp_r, beams.qp_r_prime,
            beams.qp_u_dot, beams.qp_omega, beams.qp_u_ddot, beams.qp_omega_dot, beams.qp_x
        }
    );

    return beams;
}

}  // namespace openturbine
