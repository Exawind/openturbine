#include "beams.hpp"

#include "interpolation.h"

namespace oturb {

void PopulateElementViews(
    const BeamInput& input, View_Nx7 node_x0, View_Nx7 node_u, View_Nx6 node_u_dot,
    View_Nx6 node_u_ddot, View_N qp_weight, View_Nx6x6 qp_Mstar, View_Nx6x6 qp_Cstar,
    View_NxN shape_interp, View_NxN shape_deriv
) {
    //--------------------------------------------------------------------------
    // Calculate element's node and quadrature point positions [-1,1]
    //--------------------------------------------------------------------------

    std::vector<double> node_xi(input.nodes.size());

    // Loop through nodes and convert 's' [0,1] position to xi
    for (size_t i = 0; i < input.nodes.size(); ++i) {
        node_xi[i] = 2 * input.nodes[i].s_ - 1;
    }

    //--------------------------------------------------------------------------
    // Populate node data
    //--------------------------------------------------------------------------

    // Loop through nodes
    for (size_t j = 0; j < input.nodes.size(); ++j) {
        // Transfer initial position
        for (size_t k = 0; k < input.nodes[j].x_.size(); ++k) {
            node_x0(j, k) = input.nodes[j].x_[k];
        }

        // Transfer initial displacement
        for (size_t k = 0; k < input.nodes[j].q_.size(); ++k) {
            node_u(j, k) = input.nodes[j].q_[k];
        }

        // Transfer initial velocity
        for (size_t k = 0; k < input.nodes[j].v_.size(); ++k) {
            node_u_dot(j, k) = input.nodes[j].v_[k];
        }

        // Transfer initial acceleration
        for (size_t k = 0; k < input.nodes[j].a_.size(); ++k) {
            node_u_ddot(j, k) = input.nodes[j].a_[k];
        }
    }

    //--------------------------------------------------------------------------
    // Populate quadrature point weights
    //--------------------------------------------------------------------------

    // Get vector of quadrature weights
    auto qp_w = input.quadrature.GetQuadratureWeights();

    for (size_t j = 0; j < qp_w.size(); ++j) {
        qp_weight(j) = qp_w[j];
    }

    //--------------------------------------------------------------------------
    // Populate shape function interpolation/derivative matrices
    //--------------------------------------------------------------------------

    std::vector<double> weights;

    // Get vector of quadrature points
    auto qp_xi = input.quadrature.GetQuadraturePoints();

    // Loop through quadrature points
    for (size_t j = 0; j < qp_xi.size(); ++j) {
        // Get interpolation weights to go from nodes to this QP
        LagrangePolynomialInterpWeights(qp_xi[j], node_xi, weights);

        // Copy interp weights to host matrix
        for (size_t k = 0; k < node_xi.size(); ++k) {
            shape_interp(k, j) = weights[k];
        }

        // Get derivative weights to go from nodes to this QP
        LagrangePolynomialDerivWeights(qp_xi[j], node_xi, weights);

        // Copy deriv weights to host matrix
        for (size_t k = 0; k < node_xi.size(); ++k) {
            shape_deriv(k, j) = weights[k];
        }
    }

    //--------------------------------------------------------------------------
    // Interpolate section mass and stiffness matrices to quadrature points
    // TODO: remove assumption that s runs from 0 to 1.
    //--------------------------------------------------------------------------

    // Get section positions [-1,1]
    std::vector<double> section_xi(input.sections.size());

    // Loop through sections and convert 's' [0,1] position to xi
    for (size_t i = 0; i < input.sections.size(); ++i) {
        section_xi[i] = 2 * input.sections[i].s - 1;
    }

    // Fill view with zeros
    Kokkos::deep_copy(qp_Mstar, 0.);
    Kokkos::deep_copy(qp_Cstar, 0.);

    // Loop through quadrature points and calculate section weights for interp
    for (size_t i = 0; i < qp_xi.size(); ++i) {
        // Calculate weights
        LagrangePolynomialInterpWeights(qp_xi[i], section_xi, weights);

        // Loop through sections
        for (size_t j = 0; j < section_xi.size(); ++j) {
            for (size_t m = 0; m < 6; ++m) {
                for (size_t n = 0; n < 6; ++n) {
                    qp_Mstar(i, m, n) += input.sections[j].M_star[m][n] * weights[j];
                    qp_Cstar(i, m, n) += input.sections[j].C_star[m][n] * weights[j];
                }
            }
        }
    }
}

Beams InitializeBeams(std::vector<BeamInput> elem_inputs) {
    //--------------------------------------------------------------------------
    // Calc number of beams, nodes, quadrature points for sizing Beams views
    //--------------------------------------------------------------------------

    // Calculate sizes for initializing Beams structure
    size_t total_beams = elem_inputs.size();
    size_t total_nodes = 0;
    size_t total_qps = 0;
    size_t max_elem_qps = 0;
    for (const auto& beam_input : elem_inputs) {
        auto num_elem_nodes = beam_input.nodes.size();
        auto num_elem_qps = beam_input.quadrature.GetNumberOfQuadraturePoints();
        total_nodes += num_elem_nodes;
        total_qps += num_elem_qps;
        max_elem_qps = std::max(max_elem_qps, num_elem_qps);
    }

    //--------------------------------------------------------------------------
    // Create Beams structure with views sized
    //--------------------------------------------------------------------------

    // Create beams structure on device
    auto beams = Beams(total_beams, total_nodes, total_qps, max_elem_qps);

    //--------------------------------------------------------------------------
    // Create host mirror of relevant views for initialization from input data
    //--------------------------------------------------------------------------

    auto host_elem_indices = Kokkos::create_mirror(beams.elem_indices);
    auto host_node_x0 = Kokkos::create_mirror(beams.node_x0);
    auto host_node_u = Kokkos::create_mirror(beams.node_u);
    auto host_node_u_dot = Kokkos::create_mirror(beams.node_u_dot);
    auto host_node_u_ddot = Kokkos::create_mirror(beams.node_u_ddot);

    auto host_qp_weight = Kokkos::create_mirror(beams.qp_weight);
    auto host_qp_Mstar = Kokkos::create_mirror(beams.qp_Mstar);
    auto host_qp_Cstar = Kokkos::create_mirror(beams.qp_Cstar);

    auto host_shape_interp = Kokkos::create_mirror(beams.shape_interp);
    auto host_shape_deriv = Kokkos::create_mirror(beams.shape_deriv);

    //--------------------------------------------------------------------------
    // Populate indices and element host views from input data
    //--------------------------------------------------------------------------

    int node_counter = 0;
    int qp_counter = 0;

    // Loop through element inputs
    for (size_t i = 0; i < elem_inputs.size(); i++) {
        // Calculate node indices for this element
        int num_nodes = elem_inputs[i].nodes.size();
        auto node_range = Kokkos::make_pair(node_counter, node_counter + num_nodes);
        node_counter += num_nodes;

        // Calculate quadrature point indices for this element
        int num_qps = elem_inputs[i].quadrature.GetNumberOfQuadraturePoints();
        auto qp_range = Kokkos::make_pair(qp_counter, qp_counter + num_qps);
        qp_counter += num_qps;

        // Populate element index
        host_elem_indices[i].num_nodes = num_nodes;
        host_elem_indices[i].node_range = node_range;
        host_elem_indices[i].num_qps = num_qps;
        host_elem_indices[i].qp_range = qp_range;

        // Populate views for this element
        PopulateElementViews(
            elem_inputs[i],  // Element inputs
            Kokkos::subview(host_node_x0, node_range, Kokkos::ALL),
            Kokkos::subview(host_node_u, node_range, Kokkos::ALL),
            Kokkos::subview(host_node_u_dot, node_range, Kokkos::ALL),
            Kokkos::subview(host_node_u_ddot, node_range, Kokkos::ALL),
            Kokkos::subview(host_qp_weight, qp_range),
            Kokkos::subview(host_qp_Mstar, qp_range, Kokkos::ALL, Kokkos::ALL),
            Kokkos::subview(host_qp_Cstar, qp_range, Kokkos::ALL, Kokkos::ALL),
            Kokkos::subview(host_shape_interp, node_range, Kokkos::make_pair(0, num_qps)),
            Kokkos::subview(host_shape_deriv, node_range, Kokkos::make_pair(0, num_qps))
        );
    }

    //--------------------------------------------------------------------------
    // Copy host data to beam views
    //--------------------------------------------------------------------------

    // Copy from host to device
    Kokkos::deep_copy(beams.elem_indices, host_elem_indices);
    Kokkos::deep_copy(beams.node_x0, host_node_x0);
    Kokkos::deep_copy(beams.node_u, host_node_u);
    Kokkos::deep_copy(beams.node_u_dot, host_node_u_dot);
    Kokkos::deep_copy(beams.node_u_ddot, host_node_u_ddot);
    Kokkos::deep_copy(beams.qp_weight, host_qp_weight);
    Kokkos::deep_copy(beams.qp_Mstar, host_qp_Mstar);
    Kokkos::deep_copy(beams.qp_Cstar, host_qp_Cstar);
    Kokkos::deep_copy(beams.shape_interp, host_shape_interp);
    Kokkos::deep_copy(beams.shape_deriv, host_shape_deriv);

    //--------------------------------------------------------------------------
    // Perform parallel initialization of data
    //--------------------------------------------------------------------------

    // Set state index for each node
    // TODO: update for assembly where state may apply to multiple nodes in different elements
    Kokkos::parallel_for(
        "SetNodeStateIndices", beams.node_u.extent(0),
        KOKKOS_LAMBDA(size_t i) { beams.node_state_indices(i) = i; }
    );

    // Interpolate node positions to quadrature points
    Kokkos::parallel_for(
        "InterpolateQPPosition", beams.num,
        InterpolateQPPosition{beams.elem_indices, beams.shape_interp, beams.node_x0, beams.qp_x0}
    );

    // Interpolate node rotations to quadrature points
    Kokkos::parallel_for(
        "InterpolateQPRotation", beams.num,
        InterpolateQPRotation{beams.elem_indices, beams.shape_interp, beams.node_x0, beams.qp_r0}
    );

    // Calculate derivative of position (Jacobian) and x0_prime
    Kokkos::parallel_for(
        "CalculateJacobian", beams.num,
        CalculateJacobian{
            beams.elem_indices,
            beams.shape_deriv,
            beams.node_x0,
            beams.qp_x0_prime,
            beams.qp_jacobian,
        }
    );

    // Interpolate initial state to quadrature points
    Kokkos::parallel_for(
        "InterpolateQPState", beams.num,
        InterpolateQPState{
            beams.elem_indices, beams.shape_interp, beams.shape_deriv, beams.qp_jacobian,
            beams.node_u, beams.node_u_dot, beams.node_u_ddot, beams.qp_u, beams.qp_u_prime,
            beams.qp_r, beams.qp_r_prime, beams.qp_u_dot, beams.qp_omega, beams.qp_u_ddot,
            beams.qp_omega_dot}
    );

    return beams;
}

}  // namespace oturb