#pragma once

#include <Kokkos_Core.hpp>

#include "src/restruct_poc/beams/beams.hpp"
#include "src/restruct_poc/types.hpp"

namespace openturbine {

struct CalculateNodeForces_FE {
    size_t i_elem;
    size_t first_node;
    size_t first_qp;
    size_t num_qps;
    Kokkos::View<double**>::const_type qp_weight_;
    Kokkos::View<double**>::const_type qp_jacobian_;
    Kokkos::View<double***>::const_type shape_interp_;
    Kokkos::View<double***>::const_type shape_deriv_;
    Kokkos::View<double** [6]>::const_type qp_Fc_;
    Kokkos::View<double** [6]>::const_type qp_Fd_;
    Kokkos::View<double** [6]> node_FE_;

    KOKKOS_FUNCTION
    void operator()(size_t i_index) const {
        auto FE = Kokkos::Array<double, 6>{};
        for (auto j_index = 0U; j_index < num_qps; ++j_index) {  // QPs
            // const auto j = first_qp + j_index;
            const auto weight = qp_weight_(i_elem, j_index);
            const auto coeff_c = weight * shape_deriv_(i_elem, i_index, j_index);
            const auto coeff_d =
                weight * qp_jacobian_(i_elem, j_index) * shape_interp_(i_elem, i_index, j_index);
            for (auto k = 0U; k < 6U; ++k) {
                FE[k] += coeff_c * qp_Fc_(i_elem, j_index, k) + coeff_d * qp_Fd_(i_elem, j_index, k);
            }
        }
        for (auto k = 0U; k < 6U; ++k) {
            node_FE_(i_elem, i_index, k) = FE[k];
        }
    }
};

struct CalculateNodeForces_FI_FG {
    size_t i_elem;
    size_t first_node;
    size_t first_qp;
    size_t num_qps;
    Kokkos::View<double**>::const_type qp_weight_;
    Kokkos::View<double**>::const_type qp_jacobian_;
    Kokkos::View<double***>::const_type shape_interp_;
    Kokkos::View<double***>::const_type shape_deriv_;
    Kokkos::View<double** [6]>::const_type qp_Fig_;
    Kokkos::View<double** [6]> node_FIG_;

    KOKKOS_FUNCTION
    void operator()(size_t i_index) const {
        auto FIG = Kokkos::Array<double, 6>{};
        for (auto j_index = 0U; j_index < num_qps; ++j_index) {  // QPs
            // const auto j = first_qp + j_index;
            const auto weight = qp_weight_(i_elem, j_index);
            const auto coeff_ig =
                weight * qp_jacobian_(i_elem, j_index) * shape_interp_(i_elem, i_index, j_index);
            for (auto k = 0U; k < 6U; ++k) {
                FIG[k] += coeff_ig * qp_Fig_(i_elem, j_index, k);
            }
        }
        for (auto k = 0U; k < 6U; ++k) {
            node_FIG_(i_elem, i_index, k) = FIG[k];
        }
    }
};

struct CalculateNodeForces {
    Kokkos::View<Beams::ElemIndices*>::const_type elem_indices;
    Kokkos::View<double**>::const_type qp_weight_;
    Kokkos::View<double**>::const_type qp_jacobian_;
    Kokkos::View<double***>::const_type shape_interp_;
    Kokkos::View<double***>::const_type shape_deriv_;
    Kokkos::View<double** [6]>::const_type qp_Fc_;
    Kokkos::View<double** [6]>::const_type qp_Fd_;
    Kokkos::View<double** [6]>::const_type qp_Fi_;
    Kokkos::View<double** [6]>::const_type qp_Fg_;
    Kokkos::View<double** [6]> node_FE_;
    Kokkos::View<double** [6]> node_FI_;
    Kokkos::View<double** [6]> node_FG_;

    KOKKOS_FUNCTION
    void operator()(Kokkos::TeamPolicy<>::member_type member) const {
        const auto i_elem = static_cast<size_t>(member.league_rank());
        const auto idx = elem_indices(i_elem);
        const auto first_node = idx.node_range.first;
        const auto first_qp = idx.qp_range.first;
        const auto num_qps = idx.num_qps;

        Kokkos::parallel_for(
            Kokkos::TeamThreadRange(member, idx.num_nodes),
            CalculateNodeForces_FE{
                i_elem, first_node, first_qp, num_qps, qp_weight_, qp_jacobian_, shape_interp_,
                shape_deriv_, qp_Fc_, qp_Fd_, node_FE_}
        );
        Kokkos::parallel_for(
            Kokkos::TeamThreadRange(member, idx.num_nodes),
            CalculateNodeForces_FI_FG{
                i_elem, first_node, first_qp, num_qps, qp_weight_, qp_jacobian_, shape_interp_,
                shape_deriv_, qp_Fi_, node_FI_}
        );
        Kokkos::parallel_for(
            Kokkos::TeamThreadRange(member, idx.num_nodes),
            CalculateNodeForces_FI_FG{
                i_elem, first_node, first_qp, num_qps, qp_weight_, qp_jacobian_, shape_interp_,
                shape_deriv_, qp_Fg_, node_FG_}
        );
    }
};
}  // namespace openturbine
