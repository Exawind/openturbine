#pragma once

#include <vector>

#include "interpolation.hpp"

namespace openturbine {

template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6>
inline void PopulateElementViews(
    const BeamElement& elem, T1 node_x0, T2 qp_weight, T3 qp_Mstar, T4 qp_Cstar, T5 shape_interp,
    T6 shape_deriv
) {
    std::vector<double> node_xi(elem.nodes.size());
    for (size_t i = 0; i < elem.nodes.size(); ++i) {
        node_xi[i] = 2 * elem.nodes[i].position - 1;
    }

    for (size_t j = 0; j < elem.nodes.size(); ++j) {
        for (size_t k = 0; k < elem.nodes[j].initial_dofs.size(); ++k) {
            node_x0(j, k) = elem.nodes[j].initial_dofs[k];
        }
    }

    for (size_t j = 0; j < elem.quadrature.size(); ++j) {
        qp_weight(j) = elem.quadrature[j][1];
    }

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

    // TODO: remove assumption that s runs from 0 to 1
    std::vector<double> section_xi(elem.sections.size());

    for (size_t i = 0; i < elem.sections.size(); ++i) {
        section_xi[i] = 2 * elem.sections[i].s - 1;
    }

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
