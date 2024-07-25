#pragma once

#include <Kokkos_Core.hpp>

#include "src/restruct_poc/beams/beams.hpp"
#include "src/restruct_poc/types.hpp"

namespace openturbine {

struct CalculateNodeForces_FE {
    size_t first_node;
    size_t first_qp;
    size_t num_qps;
    View_N::const_type qp_weight_;
    View_N::const_type qp_jacobian_;
    View_NxN::const_type shape_interp_;
    View_NxN::const_type shape_deriv_;
    View_Nx6::const_type qp_Fc_;
    View_Nx6::const_type qp_Fd_;
    View_Nx6 node_FE_;

    KOKKOS_FUNCTION
    void operator()(size_t i_index) const {
        const auto i = first_node + i_index;
        auto FE = Kokkos::Array<double, 6>{};
        for (auto j_index = 0u; j_index < num_qps; ++j_index) {  // QPs
            const auto j = first_qp + j_index;
            const auto weight = qp_weight_(j);
            const auto coeff_c = weight * shape_deriv_(i, j_index);
            const auto coeff_d = weight * qp_jacobian_(j) * shape_interp_(i, j_index);
            for (auto k = 0u; k < 6u; ++k) {
                FE[k] += coeff_c * qp_Fc_(j, k) + coeff_d * qp_Fd_(j, k);
            }
        }
        for (auto k = 0u; k < 6u; ++k) {
            node_FE_(i, k) = FE[k];
        }
    }
};

struct CalculateNodeForces_FI_FG {
    size_t first_node;
    size_t first_qp;
    size_t num_qps;
    View_N::const_type qp_weight_;
    View_N::const_type qp_jacobian_;
    View_NxN::const_type shape_interp_;
    View_NxN::const_type shape_deriv_;
    View_Nx6::const_type qp_Fig_;
    View_Nx6 node_FIG_;

    KOKKOS_FUNCTION
    void operator()(size_t i_index) const {
        const auto i = first_node + i_index;
        auto FIG = Kokkos::Array<double, 6>{};
        for (auto j_index = 0u; j_index < num_qps; ++j_index) {  // QPs
            const auto j = first_qp + j_index;
            const auto weight = qp_weight_(j);
            const auto coeff_ig = weight * qp_jacobian_(j) * shape_interp_(i, j_index);
            for (auto k = 0u; k < 6u; ++k) {
                FIG[k] += coeff_ig * qp_Fig_(j, k);
            }
        }
        for (auto k = 0u; k < 6u; ++k) {
            node_FIG_(i, k) = FIG[k];
        }
    }
};

struct CalculateNodeForces {
    Kokkos::View<Beams::ElemIndices*>::const_type elem_indices;
    View_N::const_type qp_weight_;
    View_N::const_type qp_jacobian_;
    View_NxN::const_type shape_interp_;
    View_NxN::const_type shape_deriv_;
    View_Nx6::const_type qp_Fc_;
    View_Nx6::const_type qp_Fd_;
    View_Nx6::const_type qp_Fi_;
    View_Nx6::const_type qp_Fg_;
    View_Nx6 node_FE_;
    View_Nx6 node_FI_;
    View_Nx6 node_FG_;

    KOKKOS_FUNCTION
    void operator()(Kokkos::TeamPolicy<>::member_type member) const {
        const auto idx = elem_indices(member.league_rank());
        const auto first_node = idx.node_range.first;
        const auto first_qp = idx.qp_range.first;
        const auto num_qps = idx.num_qps;

        Kokkos::parallel_for(
            Kokkos::TeamThreadRange(member, idx.num_nodes),
            CalculateNodeForces_FE{
                first_node, first_qp, num_qps, qp_weight_, qp_jacobian_, shape_interp_, shape_deriv_,
                qp_Fc_, qp_Fd_, node_FE_}
        );
        Kokkos::parallel_for(
            Kokkos::TeamThreadRange(member, idx.num_nodes),
            CalculateNodeForces_FI_FG{
                first_node, first_qp, num_qps, qp_weight_, qp_jacobian_, shape_interp_, shape_deriv_,
                qp_Fi_, node_FI_}
        );
        Kokkos::parallel_for(
            Kokkos::TeamThreadRange(member, idx.num_nodes),
            CalculateNodeForces_FI_FG{
                first_node, first_qp, num_qps, qp_weight_, qp_jacobian_, shape_interp_, shape_deriv_,
                qp_Fg_, node_FG_}
        );
    }
};
}  // namespace openturbine
