#pragma once

#include <Kokkos_Core.hpp>

#include "src/restruct_poc/beams/beams.hpp"
#include "src/restruct_poc/types.hpp"

namespace openturbine {

struct IntegrateInertiaMatrixElement {
    int i_elem;
    size_t num_qps;
    size_t first_node;
    size_t first_qp;
    View_N::const_type qp_weight_;
    View_N::const_type qp_jacobian_;
    View_NxN::const_type shape_interp_;
    View_Nx6x6::const_type qp_Muu_;
    View_Nx6x6::const_type qp_Guu_;
    double beta_prime_;
    double gamma_prime_;
    Kokkos::View<double***> gbl_M_;

    KOKKOS_FUNCTION
    void operator()(size_t i_index, size_t j_index) const {
        const auto i = i_index + first_node;
        const auto j = j_index + first_node;
        auto local_M_data = Kokkos::Array<double, 36>{};
        const auto local_M = Kokkos::View<double[6][6]>(local_M_data.data());
        for (auto k = 0U; k < num_qps; ++k) {
            const auto k_qp = first_qp + k;
            const auto w = qp_weight_(k_qp);
            const auto jacobian = qp_jacobian_(k_qp);
            const auto phi_i = shape_interp_(i, k);
            const auto phi_j = shape_interp_(j, k);
            const auto coeff = w * phi_i * phi_j * jacobian;
            for (auto m = 0U; m < 6U; ++m) {
                for (auto n = 0U; n < 6U; ++n) {
                    local_M(m, n) += beta_prime_ * coeff * qp_Muu_(k_qp, m, n) +
                                     gamma_prime_ * coeff * qp_Guu_(k_qp, m, n);
                }
            }
        }
        for (auto m = 0U; m < 6U; ++m) {
            for (auto n = 0U; n < 6U; ++n) {
                gbl_M_(i_elem, i_index * 6 + m, j_index * 6 + n) = local_M(m, n);
            }
        }
    }
};
struct IntegrateInertiaMatrix {
    Kokkos::View<Beams::ElemIndices*>::const_type elem_indices;
    View_N::const_type qp_weight_;
    View_N::const_type qp_jacobian_;
    View_NxN::const_type shape_interp_;
    View_Nx6x6::const_type qp_Muu_;
    View_Nx6x6::const_type qp_Guu_;
    double beta_prime_;
    double gamma_prime_;
    Kokkos::View<double***> gbl_M_;

    KOKKOS_FUNCTION
    void operator()(const Kokkos::TeamPolicy<>::member_type& member) const {
        const auto i_elem = member.league_rank();
        const auto idx = elem_indices(i_elem);
        const auto node_range = Kokkos::TeamThreadMDRange(member, idx.num_nodes, idx.num_nodes);
        const auto element_integrator = IntegrateInertiaMatrixElement{
            i_elem,     idx.num_qps,  idx.node_range.first, idx.qp_range.first,
            qp_weight_, qp_jacobian_, shape_interp_,        qp_Muu_,
            qp_Guu_,    beta_prime_,  gamma_prime_,         gbl_M_};
        Kokkos::parallel_for(node_range, element_integrator);
    }
};
}  // namespace openturbine
