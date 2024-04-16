#pragma once

#include <Kokkos_Core.hpp>

#include "types.hpp"
#include "beams_data.hpp"

namespace openturbine {

struct CalculateNodeForces {
    Kokkos::View<Beams::ElemIndices*>::const_type elem_indices;  // Element indices
    View_N::const_type qp_weight_;                               //
    View_N::const_type qp_jacobian_;                             // Jacobians
    View_NxN::const_type shape_interp_;                          // Num Nodes x Num Quadrature points
    View_NxN::const_type shape_deriv_;                           // Num Nodes x Num Quadrature points
    View_Nx6::const_type qp_Fc_;                                 //
    View_Nx6::const_type qp_Fd_;                                 //
    View_Nx6::const_type qp_Fi_;                                 //
    View_Nx6::const_type qp_Fg_;                                 //
    View_Nx6 node_FE_;                                           // Elastic force
    View_Nx6 node_FI_;                                           // Inertial force
    View_Nx6 node_FG_;                                           // Gravity force

    KOKKOS_FUNCTION
    void operator()(const int i_elem) const {
        auto idx = elem_indices[i_elem];

        for (int i = idx.node_range.first; i < idx.node_range.second; ++i) {
            for (int j = 0; j < 6; ++j) {
                node_FE_(i, j) = 0.;
                node_FG_(i, j) = 0.;
                node_FI_(i, j) = 0.;
            }
        }

        for (int i_index = 0; i_index < idx.num_nodes; ++i_index) {    // Nodes
            for (int j_index = 0; j_index < idx.num_qps; ++j_index) {  // QPs
                const auto i = idx.node_range.first + i_index;
                const auto j = idx.qp_range.first + j_index;
                const auto weight = qp_weight_(j);
                const auto coeff_c = weight * shape_deriv_(i, j_index);
                const auto coeff_d = weight * qp_jacobian_(j) * shape_interp_(i, j_index);
                const auto coeff_i = coeff_d;
                const auto coeff_g = coeff_d;
                for (int k = 0; k < 6; ++k) {  // Components
                    node_FE_(i, k) += coeff_c * qp_Fc_(j, k) + coeff_d * qp_Fd_(j, k);
                }
                for (int k = 0; k < 6; ++k) {  // Components
                    node_FI_(i, k) += coeff_i * qp_Fi_(j, k);
                }
                for (int k = 0; k < 6; ++k) {  // Components
                    node_FG_(i, k) += coeff_g * qp_Fg_(j, k);
                }
            }
        }
    }

    KOKKOS_FUNCTION
    void operator()(Kokkos::TeamPolicy<>::member_type member) const {
        const auto idx = elem_indices(member.league_rank());
        Kokkos::parallel_for(
            Kokkos::TeamThreadMDRange(member, idx.num_nodes, 6),
            [=](int i_index, int j) {
                const auto i = idx.node_range.first + i_index;
                node_FE_(i, j) = 0.;
                node_FG_(i, j) = 0.;
                node_FI_(i, j) = 0.;
            }
        );

        Kokkos::parallel_for(Kokkos::TeamThreadRange(member, idx.num_nodes), [=](int i_index) {
            for (int j_index = 0; j_index < idx.num_qps; ++j_index) {  // QPs
                const auto i = idx.node_range.first + i_index;
                const auto j = idx.qp_range.first + j_index;
                const auto weight = qp_weight_(j);
                const auto coeff_c = weight * shape_deriv_(i, j_index);
                const auto coeff_d = weight * qp_jacobian_(j) * shape_interp_(i, j_index);
                const auto coeff_i = coeff_d;
                const auto coeff_g = coeff_d;
                Kokkos::parallel_for(Kokkos::ThreadVectorRange(member, 6), [=](int k) {
                    node_FE_(i, k) += coeff_c * qp_Fc_(j, k) + coeff_d * qp_Fd_(j, k);
                });
                Kokkos::parallel_for(Kokkos::ThreadVectorRange(member, 6), [=](int k) {
                    node_FI_(i, k) += coeff_i * qp_Fi_(j, k);
                });
                Kokkos::parallel_for(Kokkos::ThreadVectorRange(member, 6), [=](int k) {
                    node_FG_(i, k) += coeff_g * qp_Fg_(j, k);
                });
            }
        });
    }

    KOKKOS_FUNCTION
    void operator()(const int i_elem, const int i_index) const {
        const auto idx = elem_indices(i_elem);
        const auto i = idx.node_range.first + i_index;

        if (i_index >= idx.num_nodes) {
            return;
        }

        auto local_FE = Kokkos::Array<double, 6>{};
        auto local_FI = Kokkos::Array<double, 6>{};
        auto local_FG = Kokkos::Array<double, 6>{};

        for (int j_index = 0; j_index < idx.num_qps; ++j_index) {
            const auto j = idx.qp_range.first + j_index;
            const auto weight = qp_weight_(j);
            const auto coeff_c = weight * shape_deriv_(i, j_index);
            const auto coeff_d = weight * qp_jacobian_(j) * shape_interp_(i, j_index);
            const auto coeff_i = coeff_d;
            const auto coeff_g = coeff_d;
            for (int k = 0; k < 6; ++k) {
                local_FE[k] += coeff_c * qp_Fc_(j, k) + coeff_d * qp_Fd_(j, k);
                local_FI[k] += coeff_i * qp_Fi_(j, k);
                local_FG[k] += coeff_g * qp_Fg_(j, k);
            }
        }

        for (int k = 0; k < 6; ++k) {
            node_FE_(i, k) = local_FE[k];
            node_FI_(i, k) = local_FI[k];
            node_FG_(i, k) = local_FG[k];
        }
    }
};

struct CalculateNodeForces_FE {
    Kokkos::View<Beams::ElemIndices*>::const_type elem_indices;  // Element indices
    View_N::const_type qp_weight_;                               //
    View_N::const_type qp_jacobian_;                             // Jacobians
    View_NxN::const_type shape_interp_;                          // Num Nodes x Num Quadrature points
    View_NxN::const_type shape_deriv_;                           // Num Nodes x Num Quadrature points
    View_Nx6::const_type qp_Fc_;                                 //
    View_Nx6::const_type qp_Fd_;                                 //
    View_Nx6 node_FE_;                                           // Elastic force

    KOKKOS_FUNCTION
    void operator()(const int i_elem, const int i_index) const {
        const auto idx = elem_indices(i_elem);
        const auto i = idx.node_range.first + i_index;

        if (i_index >= idx.num_nodes) {
            return;
        }

        auto local_FE = Kokkos::Array<double, 6>{};
        for (int j_index = 0; j_index < idx.num_qps; ++j_index) {
            const auto j = idx.qp_range.first + j_index;
            const auto weight = qp_weight_(j);
            const auto coeff_c = weight * shape_deriv_(i, j_index);
            const auto coeff_d = weight * qp_jacobian_(j) * shape_interp_(i, j_index);
            for (int k = 0; k < 6; ++k) {
                local_FE[k] += coeff_c * qp_Fc_(j, k) + coeff_d * qp_Fd_(j, k);
            }
        }
        for (int k = 0; k < 6; ++k) {
            node_FE_(i, k) = local_FE[k];
        }
    }
};

struct CalculateNodeForces_FI {
    Kokkos::View<Beams::ElemIndices*>::const_type elem_indices;  // Element indices
    View_N::const_type qp_weight_;                               //
    View_N::const_type qp_jacobian_;                             // Jacobians
    View_NxN::const_type shape_interp_;                          // Num Nodes x Num Quadrature points
    View_NxN::const_type shape_deriv_;                           // Num Nodes x Num Quadrature points
    View_Nx6::const_type qp_Fi_;                                 //
    View_Nx6 node_FI_;                                           // Elastic force

    KOKKOS_FUNCTION
    void operator()(const int i_elem, const int i_index) const {
        const auto idx = elem_indices(i_elem);
        const auto i = idx.node_range.first + i_index;

        if (i_index >= idx.num_nodes) {
            return;
        }

        auto local_FI = Kokkos::Array<double, 6>{};
        for (int j_index = 0; j_index < idx.num_qps; ++j_index) {
            const auto j = idx.qp_range.first + j_index;
            const auto weight = qp_weight_(j);
            const auto coeff_i = weight * qp_jacobian_(j) * shape_interp_(i, j_index);
            for (int k = 0; k < 6; ++k) {
                local_FI[k] += coeff_i * qp_Fi_(j, k);
            }
        }
        for (int k = 0; k < 6; ++k) {
            node_FI_(i, k) = local_FI[k];
        }
    }
};

struct CalculateNodeForces_FG {
    Kokkos::View<Beams::ElemIndices*>::const_type elem_indices;  // Element indices
    View_N::const_type qp_weight_;                               //
    View_N::const_type qp_jacobian_;                             // Jacobians
    View_NxN::const_type shape_interp_;                          // Num Nodes x Num Quadrature points
    View_NxN::const_type shape_deriv_;                           // Num Nodes x Num Quadrature points
    View_Nx6::const_type qp_Fg_;                                 //
    View_Nx6 node_FG_;                                           // Elastic force

    KOKKOS_FUNCTION
    void operator()(const int i_elem, const int i_index) const {
        const auto idx = elem_indices(i_elem);
        const auto i = idx.node_range.first + i_index;
        if (i_index >= idx.num_nodes) {
            return;
        }

        auto local_FG = Kokkos::Array<double, 6>{};
        for (int j_index = 0; j_index < idx.num_qps; ++j_index) {
            const auto j = idx.qp_range.first + j_index;
            const auto weight = qp_weight_(j);
            const auto coeff_g = weight * qp_jacobian_(j) * shape_interp_(i, j_index);
            for (int k = 0; k < 6; ++k) {
                local_FG[k] += coeff_g * qp_Fg_(j, k);
            }
        }
        for (int k = 0; k < 6; ++k) {
            node_FG_(i, k) = local_FG[k];
        }
    }
};
}
