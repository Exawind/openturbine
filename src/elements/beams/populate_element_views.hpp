#pragma once

#include <vector>

#include "beam_element.hpp"
#include "interpolation.hpp"

namespace openturbine {

inline void PopulateNodeX0(
    const BeamElement& elem, const std::vector<Node>& nodes,
    const Kokkos::View<double* [7], Kokkos::LayoutStride, Kokkos::HostSpace>& node_x0
) {
    for (size_t j = 0; j < elem.node_ids.size(); ++j) {
        for (size_t k = 0U; k < kLieGroupComponents; ++k) {
            node_x0(j, k) = nodes[elem.node_ids[j]].x[k];
        }
    }
}

inline void PopulateQPWeight(
    const BeamElement& elem,
    const Kokkos::View<double*, Kokkos::LayoutStride, Kokkos::HostSpace>& qp_weight
) {
    for (size_t j = 0; j < elem.quadrature.size(); ++j) {
        qp_weight(j) = elem.quadrature[j][1];
    }
}

inline void PopulateShapeFunctionValues(
    const BeamElement& elem, const std::vector<Node>& nodes,
    const Kokkos::View<double**, Kokkos::LayoutStride, Kokkos::HostSpace>& shape_interp
) {
    std::vector<double> node_xi(elem.node_ids.size());
    for (size_t i = 0; i < elem.node_ids.size(); ++i) {
        node_xi[i] = 2 * nodes[elem.node_ids[i]].s - 1;
    }

    std::vector<double> weights;
    for (size_t j = 0; j < elem.quadrature.size(); ++j) {
        auto qp_xi = elem.quadrature[j][0];

        LagrangePolynomialInterpWeights(qp_xi, node_xi, weights);
        for (size_t k = 0; k < node_xi.size(); ++k) {
            shape_interp(k, j) = weights[k];
        }
    }
}

inline void PopulateShapeFunctionDerivatives(
    const BeamElement& elem, const std::vector<Node>& nodes,
    const Kokkos::View<double**, Kokkos::LayoutStride, Kokkos::HostSpace>& shape_deriv
) {
    std::vector<double> node_xi(elem.node_ids.size());
    for (size_t i = 0; i < elem.node_ids.size(); ++i) {
        node_xi[i] = 2 * nodes[elem.node_ids[i]].s - 1;
    }

    std::vector<double> weights;
    for (size_t j = 0; j < elem.quadrature.size(); ++j) {
        auto qp_xi = elem.quadrature[j][0];

        LagrangePolynomialDerivWeights(qp_xi, node_xi, weights);
        for (size_t k = 0; k < node_xi.size(); ++k) {
            shape_deriv(k, j) = weights[k];
        }
    }
}

inline void PopulateQPMstar(
    const BeamElement& elem,
    const Kokkos::View<double* [6][6], Kokkos::LayoutStride, Kokkos::HostSpace>& qp_Mstar
) {
    std::vector<double> weights(elem.sections.size());
    std::vector<double> section_xi(elem.sections.size());
    for (size_t i = 0; i < elem.sections.size(); ++i) {
        section_xi[i] = 2 * elem.sections[i].position - 1;
    }

    for (size_t i = 0; i < elem.quadrature.size(); ++i) {
        auto qp_xi = elem.quadrature[i][0];
        LinearInterpWeights(qp_xi, section_xi, weights);
        for (size_t m = 0; m < 6; ++m) {
            for (size_t n = 0; n < 6; ++n) {
                qp_Mstar(i, m, n) = 0.;
            }
        }
        for (size_t j = 0; j < section_xi.size(); ++j) {
            for (size_t m = 0; m < 6; ++m) {
                for (size_t n = 0; n < 6; ++n) {
                    qp_Mstar(i, m, n) += elem.sections[j].M_star[m][n] * weights[j];
                }
            }
        }
    }
}

inline void PopulateQPCstar(
    const BeamElement& elem,
    const Kokkos::View<double* [6][6], Kokkos::LayoutStride, Kokkos::HostSpace>& qp_Cstar
) {
    std::vector<double> weights(elem.sections.size());
    std::vector<double> section_xi(elem.sections.size());
    for (size_t i = 0; i < elem.sections.size(); ++i) {
        section_xi[i] = 2 * elem.sections[i].position - 1;
    }

    for (size_t i = 0; i < elem.quadrature.size(); ++i) {
        auto qp_xi = elem.quadrature[i][0];
        LinearInterpWeights(qp_xi, section_xi, weights);
        for (size_t m = 0; m < 6; ++m) {
            for (size_t n = 0; n < 6; ++n) {
                qp_Cstar(i, m, n) = 0.;
            }
        }
        for (size_t j = 0; j < section_xi.size(); ++j) {
            for (size_t m = 0; m < 6; ++m) {
                for (size_t n = 0; n < 6; ++n) {
                    qp_Cstar(i, m, n) += elem.sections[j].C_star[m][n] * weights[j];
                }
            }
        }
    }
}
}  // namespace openturbine
