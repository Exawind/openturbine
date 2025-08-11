#pragma once

#include <vector>

#include "beam_element.hpp"
#include "math/interpolation.hpp"
#include "model/node.hpp"

namespace openturbine::beams {

/// @brief Populate the node initial position and orientation
inline void PopulateNodeX0(
    const BeamElement& elem, const std::vector<Node>& nodes,
    const Kokkos::View<double* [7], Kokkos::LayoutStride, Kokkos::HostSpace>& node_x0
) {
    for (auto node = 0U; node < elem.node_ids.size(); ++node) {
        for (auto component = 0U; component < 7U; ++component) {
            node_x0(node, component) = nodes[elem.node_ids[node]].x0[component];
        }
    }
}

/// @brief Populate the integration weights at each quadrature point
inline void PopulateQPWeight(
    const BeamElement& elem,
    const Kokkos::View<double*, Kokkos::LayoutStride, Kokkos::HostSpace>& qp_weight
) {
    for (auto qp = 0U; qp < elem.quadrature.size(); ++qp) {
        qp_weight(qp) = elem.quadrature[qp][1];
    }
}

/// @brief Map node positions from [0,1] to [-1,1]
inline std::vector<double> MapNodePositions(
    const BeamElement& elem, const std::vector<Node>& nodes
) {
    std::vector<double> node_xi(elem.node_ids.size());
    for (auto node = 0U; node < elem.node_ids.size(); ++node) {
        node_xi[node] = 2 * nodes[elem.node_ids[node]].s - 1;
    }
    return node_xi;
}

/// @brief Populate shape function values at each quadrature point
inline void PopulateShapeFunctionValues(
    const BeamElement& elem, const std::vector<Node>& nodes,
    const Kokkos::View<double**, Kokkos::LayoutStride, Kokkos::HostSpace>& shape_interp
) {
    const auto node_xi = MapNodePositions(elem, nodes);

    std::vector<double> weights;
    for (auto qp = 0U; qp < elem.quadrature.size(); ++qp) {
        auto qp_xi = elem.quadrature[qp][0];

        math::LagrangePolynomialInterpWeights(qp_xi, node_xi, weights);
        for (auto node = 0U; node < node_xi.size(); ++node) {
            shape_interp(node, qp) = weights[node];
        }
    }
}

/// @brief Populate shape function derivatives at each quadrature point
inline void PopulateShapeFunctionDerivatives(
    const BeamElement& elem, const std::vector<Node>& nodes,
    const Kokkos::View<double**, Kokkos::LayoutStride, Kokkos::HostSpace>& shape_deriv
) {
    const auto node_xi = MapNodePositions(elem, nodes);

    std::vector<double> weights;
    for (auto qp = 0U; qp < elem.quadrature.size(); ++qp) {
        auto qp_xi = elem.quadrature[qp][0];

        math::LagrangePolynomialDerivWeights(qp_xi, node_xi, weights);
        for (auto node = 0U; node < node_xi.size(); ++node) {
            shape_deriv(node, qp) = weights[node];
        }
    }
}

/// @brief Map section positions from [0,1] to [-1,1]
inline std::vector<double> MapSectionPositions(const BeamElement& elem) {
    std::vector<double> section_xi(elem.sections.size());
    for (auto section = 0U; section < elem.sections.size(); ++section) {
        section_xi[section] = 2 * elem.sections[section].position - 1;
    }
    return section_xi;
}

/// @brief Populate mass matrix values at each quadrature point
inline void PopulateQPMstar(
    const BeamElement& elem,
    const Kokkos::View<double* [6][6], Kokkos::LayoutStride, Kokkos::HostSpace>& qp_Mstar
) {
    const auto section_xi = MapSectionPositions(elem);
    std::vector<double> weights(elem.sections.size());

    for (auto qp = 0U; qp < elem.quadrature.size(); ++qp) {
        auto qp_xi = elem.quadrature[qp][0];
        math::LinearInterpWeights(qp_xi, section_xi, weights);
        for (auto component_1 = 0; component_1 < 6; ++component_1) {
            for (auto component_2 = 0; component_2 < 6; ++component_2) {
                qp_Mstar(qp, component_1, component_2) = 0.;
            }
        }
        for (auto section = 0U; section < section_xi.size(); ++section) {
            for (auto component_1 = 0U; component_1 < 6U; ++component_1) {
                for (auto component_2 = 0U; component_2 < 6U; ++component_2) {
                    qp_Mstar(qp, component_1, component_2) +=
                        elem.sections[section].M_star[component_1][component_2] * weights[section];
                }
            }
        }
    }
}

/// @brief Populate stiffness matrix values at each quadrature point
inline void PopulateQPCstar(
    const BeamElement& elem,
    const Kokkos::View<double* [6][6], Kokkos::LayoutStride, Kokkos::HostSpace>& qp_Cstar
) {
    const auto section_xi = MapSectionPositions(elem);
    std::vector<double> weights(elem.sections.size());

    for (auto qp = 0U; qp < elem.quadrature.size(); ++qp) {
        auto qp_xi = elem.quadrature[qp][0];
        math::LinearInterpWeights(qp_xi, section_xi, weights);
        for (auto component_1 = 0; component_1 < 6; ++component_1) {
            for (auto component_2 = 0; component_2 < 6; ++component_2) {
                qp_Cstar(qp, component_1, component_2) = 0.;
            }
        }
        for (auto section = 0U; section < section_xi.size(); ++section) {
            for (auto component_1 = 0U; component_1 < 6U; ++component_1) {
                for (auto component_2 = 0U; component_2 < 6U; ++component_2) {
                    qp_Cstar(qp, component_1, component_2) +=
                        elem.sections[section].C_star[component_1][component_2] * weights[section];
                }
            }
        }
    }
}
}  // namespace openturbine
