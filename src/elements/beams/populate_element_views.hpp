#pragma once

#include <vector>

#include "beam_element.hpp"
#include "interpolation.hpp"

namespace openturbine {

/**
 * @brief Populates Views with beam element data for numerical integration
 *
 * @param elem The beam element containing node, quadrature, and section data
 * @param node_x0 View for initial nodal positions and orientations (size = num_nodes x num_dofs)
 * @param qp_weight View for quadrature point weights (size = num_qps)
 * @param qp_Mstar View for interpolated mass matrices at quadrature points (size = num_qps x 6 x 6)
 * @param qp_Cstar View for interpolated stiffness matrices at quadrature points
 *                (size = num_qps x 6 x 6)
 * @param shape_interp View for shape function interpolation weights (size = num_nodes x num_qps)
 * @param shape_deriv View for shape function derivative weights (size = num_nodes x num_qps)
 *
 * This function transforms element data into a format suitable for numerical integration:
 * - Maps node and section positions from [0,1] to [-1,1]
 * - Computes shape function weights for interpolation and derivatives using Lagrange polynomials
 * - Interpolates section properties (mass and stiffness) at quadrature points using
 *   linear interpolation
 */
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6>
inline void PopulateElementViews(
    const BeamElement& elem, const std::vector<Node>& nodes, T1 node_x0, T2 qp_weight, T3 qp_Mstar,
    T4 qp_Cstar, T5 shape_interp, T6 shape_deriv
) {
    // Map node positions from [0,1] to [-1,1]
    std::vector<double> node_xi(elem.node_ids.size());
    for (size_t i = 0; i < elem.node_ids.size(); ++i) {
        node_xi[i] = 2 * nodes[elem.node_ids[i]].s - 1;
    }

    // Populate node initial position and orientation
    for (size_t j = 0; j < elem.node_ids.size(); ++j) {
        for (size_t k = 0U; k < kLieGroupComponents; ++k) {
            node_x0(j, k) = nodes[elem.node_ids[j]].x[k];
        }
    }

    // Populate quadrature weights
    for (size_t j = 0; j < elem.quadrature.size(); ++j) {
        qp_weight(j) = elem.quadrature[j][1];
    }

    // Populate shape interpolation and derivative weights (Lagrange polynomial)
    std::vector<double> weights;
    for (size_t j = 0; j < elem.quadrature.size(); ++j) {
        auto qp_xi = elem.quadrature[j][0];

        LagrangePolynomialInterpWeights(qp_xi, node_xi, weights);
        for (size_t k = 0; k < node_xi.size(); ++k) {
            shape_interp(k, j) = weights[k];
        }

        LagrangePolynomialDerivWeights(qp_xi, node_xi, weights);
        for (size_t k = 0; k < node_xi.size(); ++k) {
            shape_deriv(k, j) = weights[k];
        }
    }

    // Map section positions from [0,1] to [-1,1]
    std::vector<double> section_xi(elem.sections.size());
    for (size_t i = 0; i < elem.sections.size(); ++i) {
        section_xi[i] = 2 * elem.sections[i].position - 1;
    }

    // Populate section mass and stiffness matrices at quadrature points by
    // linearly interpolating between section values
    Kokkos::deep_copy(qp_Mstar, 0.);
    Kokkos::deep_copy(qp_Cstar, 0.);
    for (size_t i = 0; i < elem.quadrature.size(); ++i) {
        auto qp_xi = elem.quadrature[i][0];
        LinearInterpWeights(qp_xi, section_xi, weights);
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

}  // namespace openturbine
