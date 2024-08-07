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
    View_N::const_type qp_weight_;
    View_N::const_type qp_jacobian_;
    Kokkos::View<double**>::const_type shape_interp_;
    Kokkos::View<double**>::const_type shape_deriv_;
    Kokkos::View<double* [6][6]>::const_type qp_Kuu_;
    Kokkos::View<double* [6][6]>::const_type qp_Puu_;
    Kokkos::View<double* [6][6]>::const_type qp_Cuu_;
    Kokkos::View<double* [6][6]>::const_type qp_Ouu_;
    Kokkos::View<double* [6][6]>::const_type qp_Quu_;
    Kokkos::View<double***> gbl_M_;

    KOKKOS_FUNCTION
    void operator()(size_t i_index, size_t j_index) const {
        auto local_M_data = Kokkos::Array<double, 36>{};
        const auto local_M = Kokkos::View<double[6][6]>(local_M_data.data());
        for (auto k = 0U; k < num_qps; ++k) {
            const auto w = qp_weight_(k);
            const auto jacobian = qp_jacobian_(k);
            const auto phi_i = shape_interp_(i_index, k);
            const auto phi_j = shape_interp_(j_index, k);
            const auto phi_prime_i = shape_deriv_(i_index, k);
            const auto phi_prime_j = shape_deriv_(j_index, k);
            const auto K = w * phi_i * phi_j * jacobian;
            const auto P = w * (phi_i * phi_prime_j);
            const auto C = w * (phi_prime_i * phi_prime_j / jacobian);
            const auto O = w * (phi_prime_i * phi_j);
            for (auto m = 0U; m < 6U; ++m) {
                for (auto n = 0U; n < 6U; ++n) {
                    local_M(m, n) += K * (qp_Kuu_(k, m, n) + qp_Quu_(k, m, n)) +
                                     P * qp_Puu_(k, m, n) + C * qp_Cuu_(k, m, n) +
                                     O * qp_Ouu_(k, m, n);
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
    Kokkos::View<double** [6][6]>::const_type qp_Kuu_;
    Kokkos::View<double** [6][6]>::const_type qp_Puu_;
    Kokkos::View<double** [6][6]>::const_type qp_Cuu_;
    Kokkos::View<double** [6][6]>::const_type qp_Ouu_;
    Kokkos::View<double** [6][6]>::const_type qp_Quu_;
    Kokkos::View<double***> gbl_M_;

    KOKKOS_FUNCTION
    void operator()(Kokkos::TeamPolicy<>::member_type member) const {
        const auto i_elem = member.league_rank();
        const auto idx = elem_indices(i_elem);

        const auto shape_interp =
            Kokkos::View<double**>(member.team_scratch(1), idx.num_nodes, idx.num_qps);
        const auto shape_deriv =
            Kokkos::View<double**>(member.team_scratch(1), idx.num_nodes, idx.num_qps);

        const auto qp_weight = Kokkos::View<double*>(member.team_scratch(1), idx.num_qps);
        const auto qp_jacobian = Kokkos::View<double*>(member.team_scratch(1), idx.num_qps);

        const auto qp_Kuu = Kokkos::View<double* [6][6]>(member.team_scratch(1), idx.num_qps);
        const auto qp_Puu = Kokkos::View<double* [6][6]>(member.team_scratch(1), idx.num_qps);
        const auto qp_Cuu = Kokkos::View<double* [6][6]>(member.team_scratch(1), idx.num_qps);
        const auto qp_Ouu = Kokkos::View<double* [6][6]>(member.team_scratch(1), idx.num_qps);
        const auto qp_Quu = Kokkos::View<double* [6][6]>(member.team_scratch(1), idx.num_qps);

        Kokkos::parallel_for(Kokkos::TeamThreadRange(member, idx.num_qps), [=](size_t k) {
            for (auto i = 0U; i < idx.num_nodes; ++i) {
                shape_interp(i, k) = shape_interp_(i_elem, i, k);
                shape_deriv(i, k) = shape_deriv_(i_elem, i, k);
            }
            qp_weight(k) = qp_weight_(i_elem, k);
            qp_jacobian(k) = qp_jacobian_(i_elem, k);
            for (auto m = 0U; m < 6U; ++m) {
                for (auto n = 0U; n < 6U; ++n) {
                    qp_Kuu(k, m, n) = qp_Kuu_(i_elem, k, m, n);
                    qp_Puu(k, m, n) = qp_Puu_(i_elem, k, m, n);
                    qp_Cuu(k, m, n) = qp_Cuu_(i_elem, k, m, n);
                    qp_Ouu(k, m, n) = qp_Ouu_(i_elem, k, m, n);
                    qp_Quu(k, m, n) = qp_Quu_(i_elem, k, m, n);
                }
            }
        });
        member.team_barrier();

        const auto node_range = Kokkos::TeamThreadMDRange(member, idx.num_nodes, idx.num_nodes);
        const auto element_integrator = IntegrateStiffnessMatrixElement{
            i_elem,
            idx.num_qps,
            idx.node_range.first,
            idx.qp_range.first,
            qp_weight,
            qp_jacobian,
            shape_interp,
            shape_deriv,
            qp_Kuu,
            qp_Puu,
            qp_Cuu,
            qp_Ouu,
            qp_Quu,
            gbl_M_};
        Kokkos::parallel_for(node_range, element_integrator);
    }
};

}  // namespace openturbine
