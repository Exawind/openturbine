#include "beams_input.hpp"

#include "beams_functors.hpp"

namespace oturb {

void LagrangePolynomialInterpWeights(
    const double x, const std::vector<double>& xs, std::vector<double>& weights
) {
    // Get number of nodes
    auto n = xs.size();

    // Resize weights and fill with 1s
    weights.clear();
    weights.resize(n, 1.);

    // Calculate weights
    for (size_t j = 0; j < n; ++j) {
        for (size_t m = 0; m < n; ++m) {
            if (j != m) {
                weights[j] *= (x - xs[m]) / (xs[j] - xs[m]);
            }
        }
    }
}

void LagrangePolynomialDerivWeights(
    const double x, const std::vector<double>& xs, std::vector<double>& weights
) {
    // Get number of nodes
    auto n = xs.size();

    // Resize weights and fill with zeros
    weights.clear();
    weights.resize(n, 0.);

    // Calculate weights
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            if (j != i) {
                double prod = 1.0;
                for (size_t k = 0; k < n; ++k) {
                    if (k != i && k != j) {
                        prod *= (x - xs[k]) / (xs[i] - xs[k]);
                    }
                }
                weights[i] += prod / (xs[i] - xs[j]);
            }
        }
    }
}

template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6>
void PopulateElementViews(
    const BeamElement& elem, T1 node_x0, T2 qp_weight, T3 qp_Mstar, T4 qp_Cstar, T5 shape_interp,
    T6 shape_deriv
) {
    //--------------------------------------------------------------------------
    // Calculate element's node and quadrature point positions [-1,1]
    //--------------------------------------------------------------------------

    std::vector<double> node_xi(elem.nodes.size());

    // Loop through nodes and convert 's' [0,1] position to xi
    for (size_t i = 0; i < elem.nodes.size(); ++i) {
        node_xi[i] = 2 * elem.nodes[i].s - 1;
    }

    //--------------------------------------------------------------------------
    // Populate node data
    //--------------------------------------------------------------------------

    // Loop through nodes
    for (size_t j = 0; j < elem.nodes.size(); ++j) {
        // Transfer initial position
        for (size_t k = 0; k < elem.nodes[j].x.size(); ++k) {
            node_x0(j, k) = elem.nodes[j].x[k];
        }
    }

    //--------------------------------------------------------------------------
    // Populate quadrature point weights
    //--------------------------------------------------------------------------

    // Get vector of quadrature weights
    for (size_t j = 0; j < elem.quadrature.size(); ++j) {
        qp_weight(j) = elem.quadrature[j][1];
    }

    //--------------------------------------------------------------------------
    // Populate shape function interpolation/derivative matrices
    //--------------------------------------------------------------------------

    std::vector<double> weights;

    // Loop through quadrature points
    for (size_t j = 0; j < elem.quadrature.size(); ++j) {
        auto qp_xi = elem.quadrature[j][0];

        // Get interpolation weights to go from nodes to this QP
        LagrangePolynomialInterpWeights(qp_xi, node_xi, weights);

        // Copy interp weights to host matrix
        for (size_t k = 0; k < node_xi.size(); ++k) {
            shape_interp(k, j) = weights[k];
        }

        // Get derivative weights to go from nodes to this QP
        LagrangePolynomialDerivWeights(qp_xi, node_xi, weights);

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
    std::vector<double> section_xi(elem.sections.size());

    // Loop through sections and convert 's' [0,1] position to xi
    for (size_t i = 0; i < elem.sections.size(); ++i) {
        section_xi[i] = 2 * elem.sections[i].s - 1;
    }

    // Fill view with zeros
    Kokkos::deep_copy(qp_Mstar, 0.);
    Kokkos::deep_copy(qp_Cstar, 0.);

    // Loop through quadrature points and calculate section weights for interp
    for (size_t i = 0; i < elem.quadrature.size(); ++i) {
        auto qp_xi = elem.quadrature[i][0];

        // Calculate weights
        LagrangePolynomialInterpWeights(qp_xi, section_xi, weights);

        // Loop through sections
        for (size_t j = 0; j < section_xi.size(); ++j) {
            for (size_t m = 0; m < 6; ++m) {
                for (size_t n = 0; n < 6; ++n) {
                    qp_Mstar(i, m, n) += elem.sections[j].M_star[m][n] * weights[j];
                    qp_Cstar(i, m, n) += elem.sections[j].C_star[m][n] * weights[j];
                }
            }
        }
    }
}

struct SetNodeStateIndices {
    Kokkos::View<size_t*> node_state_indices;
    KOKKOS_FUNCTION
    void operator()(size_t i) const { node_state_indices(i) = i; }
};

Beams CreateBeams(const BeamsInput& beams_input) {
    Beams beams(
        beams_input.NumElements(), beams_input.NumNodes(), beams_input.NumQuadraturePoints(),
        beams_input.MaxElemQuadraturePoints()
    );

    //----------------------------------------------------------------------
    // Create host mirror of views for initialization from init data
    //----------------------------------------------------------------------

    auto host_gravity = Kokkos::create_mirror(beams.gravity);

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

    //----------------------------------------------------------------------
    // Set gravity
    //----------------------------------------------------------------------

    host_gravity(0) = beams_input.gravity[0];
    host_gravity(1) = beams_input.gravity[1];
    host_gravity(2) = beams_input.gravity[2];

    //----------------------------------------------------------------------
    // Populate indices and element host views from input data
    //----------------------------------------------------------------------

    size_t node_counter = 0;
    size_t qp_counter = 0;

    // Loop through element inputs
    for (size_t i = 0; i < beams_input.NumElements(); i++) {
        // Define node and quadrature point index data for element
        size_t num_nodes = beams_input.elements[i].nodes.size();
        size_t num_qps = beams_input.elements[i].quadrature.size();
        host_elem_indices[i] = Beams::ElemIndices(num_nodes, num_qps, node_counter, qp_counter);
        node_counter += num_nodes;
        qp_counter += num_qps;
        auto& idx = host_elem_indices[i];

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

    //----------------------------------------------------------------------
    // Copy host data to beam views
    //----------------------------------------------------------------------

    // Copy from host to device
    Kokkos::deep_copy(beams.gravity, host_gravity);
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

    //----------------------------------------------------------------------
    // Perform parallel initialization of data
    //----------------------------------------------------------------------

    // Set state index for each node
    // TODO: update for assembly where state may apply to multiple nodes in different elements
    Kokkos::parallel_for(
        "SetNodeStateIndices", beams.num_nodes, SetNodeStateIndices{beams.node_state_indices}
    );

    // Interpolate node positions to quadrature points
    Kokkos::parallel_for(
        "InterpolateQPPosition", beams.num_elems,
        InterpolateQPPosition{beams.elem_indices, beams.shape_interp, beams.node_x0, beams.qp_x0}
    );

    // Interpolate node rotations to quadrature points
    Kokkos::parallel_for(
        "InterpolateQPRotation", beams.num_elems,
        InterpolateQPRotation{beams.elem_indices, beams.shape_interp, beams.node_x0, beams.qp_r0}
    );

    // Calculate derivative of position (Jacobian) and x0_prime
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

    // Interpolate initial state to quadrature points
    Kokkos::parallel_for(
        "InterpolateQPState", beams.num_elems,
        InterpolateQPState{
            beams.elem_indices, beams.shape_interp, beams.shape_deriv, beams.qp_jacobian,
            beams.node_u, beams.qp_u, beams.qp_u_prime, beams.qp_r, beams.qp_r_prime}
    );
    Kokkos::parallel_for(
        "InterpolateQPVelocity", beams.num_elems,
        InterpolateQPVelocity{
            beams.elem_indices, beams.shape_interp, beams.shape_deriv, beams.qp_jacobian,
            beams.node_u_dot, beams.qp_u_dot, beams.qp_omega}
    );
    Kokkos::parallel_for(
        "InterpolateQPAcceleration", beams.num_elems,
        InterpolateQPAcceleration{
            beams.elem_indices, beams.shape_interp, beams.shape_deriv, beams.qp_jacobian,
            beams.node_u_ddot, beams.qp_u_ddot, beams.qp_omega_dot}
    );
    return beams;
}

}  // namespace oturb
