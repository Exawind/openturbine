#pragma once

#include <Kokkos_Core.hpp>

#include "src/restruct_poc/beams/beams.hpp"
#include "src/restruct_poc/types.hpp"

namespace openturbine {

struct IntegrateStiffnessMatrixElement {
    int i_elem;
    size_t num_qps;
    size_t first_node;
    size_t first_qp;
    View_NxN::const_type qp_weight_;
    View_NxN::const_type qp_jacobian_;
    Kokkos::View<double***>::const_type shape_interp_;
    Kokkos::View<double***>::const_type shape_deriv_;
    View_Nx6x6::const_type qp_Kuu_;
    View_Nx6x6::const_type qp_Puu_;
    View_Nx6x6::const_type qp_Cuu_;
    View_Nx6x6::const_type qp_Ouu_;
    View_Nx6x6::const_type qp_Quu_;
    Kokkos::View<double***> gbl_M_;

    KOKKOS_FUNCTION
    void operator()(size_t i_index, size_t j_index) const {
        auto local_M_data = Kokkos::Array<double, 36>{};
        const auto local_M = Kokkos::View<double[6][6]>(local_M_data.data());
        // const auto i = i_index + first_node;
        // const auto j = j_index + first_node;
        for (auto k = 0U; k < num_qps; ++k) {
            const auto k_qp = first_qp + k;
            const auto w = qp_weight_(i_elem, k);
            const auto jacobian = qp_jacobian_(i_elem, k);
            const auto phi_i = shape_interp_(i_elem, i_index, k);
            const auto phi_j = shape_interp_(i_elem, j_index, k);
            const auto phi_prime_i = shape_deriv_(i_elem, i_index, k);
            const auto phi_prime_j = shape_deriv_(i_elem, j_index, k);
            const auto K = w * phi_i * phi_j * jacobian;
            const auto P = w * (phi_i * phi_prime_j);
            const auto Q = w * phi_i * phi_j * jacobian;
            const auto C = w * (phi_prime_i * phi_prime_j / jacobian);
            const auto O = w * (phi_prime_i * phi_j);
            for (auto m = 0U; m < 6U; ++m) {
                for (auto n = 0U; n < 6U; ++n) {
                    local_M(m, n) += K * qp_Kuu_(k_qp, m, n) + P * qp_Puu_(k_qp, m, n) +
                                     Q * qp_Quu_(k_qp, m, n) + C * qp_Cuu_(k_qp, m, n) +
                                     O * qp_Ouu_(k_qp, m, n);
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

struct IntegrateStiffnessMatrix {
    Kokkos::View<Beams::ElemIndices*>::const_type elem_indices;
    View_NxN::const_type qp_weight_;
    View_NxN::const_type qp_jacobian_;
    Kokkos::View<double***>::const_type shape_interp_;
    Kokkos::View<double***>::const_type shape_deriv_;
    View_Nx6x6::const_type qp_Kuu_;
    View_Nx6x6::const_type qp_Puu_;
    View_Nx6x6::const_type qp_Cuu_;
    View_Nx6x6::const_type qp_Ouu_;
    View_Nx6x6::const_type qp_Quu_;
    Kokkos::View<double***> gbl_M_;

    KOKKOS_FUNCTION
    void operator()(Kokkos::TeamPolicy<>::member_type member) const {
        const auto i_elem = member.league_rank();
        const auto idx = elem_indices(i_elem);

        const auto node_range = Kokkos::TeamThreadMDRange(member, idx.num_nodes, idx.num_nodes);
        const auto element_integrator = IntegrateStiffnessMatrixElement{
            i_elem,
            idx.num_qps,
            idx.node_range.first,
            idx.qp_range.first,
            qp_weight_,
            qp_jacobian_,
            shape_interp_,
            shape_deriv_,
            qp_Kuu_,
            qp_Puu_,
            qp_Cuu_,
            qp_Ouu_,
            qp_Quu_,
            gbl_M_};
        Kokkos::parallel_for(node_range, element_integrator);
    }
};

}  // namespace openturbine
