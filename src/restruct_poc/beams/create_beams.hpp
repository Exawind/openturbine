#pragma once

#include "beams.hpp"
#include "calculate_jacobian.hpp"
#include "interpolate_QP_position.hpp"
#include "interpolate_QP_rotation.hpp"
#include "interpolate_to_quadrature_points.hpp"
#include "populate_element_views.hpp"
#include "set_node_state_indices.hpp"

namespace openturbine {

inline Beams CreateBeams(const BeamsInput& beams_input) {
    Beams beams(
        beams_input.NumElements(), beams_input.NumNodes(), beams_input.NumQuadraturePoints(),
        beams_input.MaxElemNodes(), beams_input.MaxElemQuadraturePoints()
    );

    auto host_gravity = Kokkos::create_mirror(beams.gravity);

    auto host_elem_indices = Kokkos::create_mirror(beams.elem_indices);
    auto host_node_state_indices = Kokkos::create_mirror(beams.node_state_indices);
    auto host_node_x0 = Kokkos::create_mirror(beams.node_x0);
    auto host_node_u = Kokkos::create_mirror(beams.node_u);
    auto host_node_u_dot = Kokkos::create_mirror(beams.node_u_dot);
    auto host_node_u_ddot = Kokkos::create_mirror(beams.node_u_ddot);

    auto host_qp_weight = Kokkos::create_mirror(beams.qp_weight);
    auto host_qp_Mstar = Kokkos::create_mirror(beams.qp_Mstar);
    auto host_qp_Cstar = Kokkos::create_mirror(beams.qp_Cstar);

    auto host_shape_interp = Kokkos::create_mirror(beams.shape_interp);
    auto host_shape_deriv = Kokkos::create_mirror(beams.shape_deriv);

    host_gravity(0) = beams_input.gravity[0];
    host_gravity(1) = beams_input.gravity[1];
    host_gravity(2) = beams_input.gravity[2];

    size_t node_counter = 0;
    size_t qp_counter = 0;

    for (size_t i = 0; i < beams_input.NumElements(); i++) {
        // Get number of nodes and quadrature points in element
        const auto num_nodes = beams_input.elements[i].nodes.size();
        const auto num_qps = beams_input.elements[i].quadrature.size();

        // Create element indices and set in host mirror
        host_elem_indices[i] = Beams::ElemIndices(num_nodes, num_qps, node_counter, qp_counter);
        const auto& idx = host_elem_indices[i];

        // Populate beam node->state indices
        for (size_t j = 0; j < num_nodes; ++j) {
            host_node_state_indices(node_counter + j) =
                static_cast<size_t>(beams_input.elements[i].nodes[j].node.ID);
        }

        // Increment counters for beam nodes and quadrature points
        node_counter += num_nodes;
        qp_counter += num_qps;

        // Populate views for this element
        PopulateElementViews(
            beams_input.elements[i],  // Element inputs
            Kokkos::subview(host_node_x0, idx.node_range, Kokkos::ALL),
            Kokkos::subview(host_qp_weight, idx.qp_range),
            Kokkos::subview(host_qp_Mstar, idx.qp_range, Kokkos::ALL, Kokkos::ALL),
            Kokkos::subview(host_qp_Cstar, idx.qp_range, Kokkos::ALL, Kokkos::ALL),
            Kokkos::subview(host_shape_interp, idx.node_range, idx.qp_shape_range),
            Kokkos::subview(host_shape_deriv, idx.node_range, idx.qp_shape_range)
        );
    }

    Kokkos::deep_copy(beams.gravity, host_gravity);
    Kokkos::deep_copy(beams.elem_indices, host_elem_indices);
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

    Kokkos::parallel_for(
        "InterpolateQPPosition", beams.num_elems,
        InterpolateQPPosition{beams.elem_indices, beams.shape_interp, beams.node_x0, beams.qp_x0}
    );

    Kokkos::parallel_for(
        "InterpolateQPRotation", beams.num_elems,
        InterpolateQPRotation{beams.elem_indices, beams.shape_interp, beams.node_x0, beams.qp_r0}
    );

    Kokkos::parallel_for(
        "CalculateJacobian", beams.num_elems,
        CalculateJacobian{
            beams.elem_indices,
            beams.shape_deriv,
            beams.node_x0,
            beams.qp_x0_prime,
            beams.qp_jacobian,
        }
    );

    auto range_policy = Kokkos::TeamPolicy<>(static_cast<int>(beams.num_elems), Kokkos::AUTO());
    Kokkos::parallel_for(
        "InterpolateToQuadraturePoints", range_policy,
        InterpolateToQuadraturePoints{
            beams.elem_indices, beams.shape_interp, beams.shape_deriv, beams.qp_jacobian,
            beams.node_u, beams.node_u_dot, beams.node_u_ddot, beams.qp_u, beams.qp_u_prime,
            beams.qp_r, beams.qp_r_prime, beams.qp_u_dot, beams.qp_omega, beams.qp_u_ddot,
            beams.qp_omega_dot}
    );
    return beams;
}

}  // namespace openturbine
