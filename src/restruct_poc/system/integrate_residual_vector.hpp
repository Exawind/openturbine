#pragma once

#include <Kokkos_Core.hpp>

#include "src/restruct_poc/beams/beams.hpp"
#include "src/restruct_poc/types.hpp"

namespace openturbine {

struct IntegrateResidualVectorElement {
    size_t i_elem;
    size_t num_qps;
    Kokkos::View<double*>::const_type qp_weight_;
    Kokkos::View<double*>::const_type qp_jacobian_;
    Kokkos::View<double**>::const_type shape_interp_;
    Kokkos::View<double**>::const_type shape_deriv_;
    Kokkos::View<double* [6]>::const_type node_FX_;
    Kokkos::View<double* [6]>::const_type qp_Fc_;
    Kokkos::View<double* [6]>::const_type qp_Fd_;
    Kokkos::View<double* [6]>::const_type qp_Fi_;
    Kokkos::View<double* [6]>::const_type qp_Fg_;
    Kokkos::View<double** [6]> residual_vector_terms_;

    KOKKOS_FUNCTION
    void operator()(size_t i_index) const {
        auto local_residual = Kokkos::Array<double, 6>{};
        for (auto j_index = 0U; j_index < num_qps; ++j_index) {  // QPs
            const auto weight = qp_weight_(j_index);
            const auto coeff_c = weight * shape_deriv_(i_index, j_index);
            const auto coeff_dig = weight * qp_jacobian_(j_index) * shape_interp_(i_index, j_index);
            for (auto k = 0U; k < 6U; ++k) {
                local_residual[k] +=
                    coeff_c * qp_Fc_(j_index, k) +
                    coeff_dig * (qp_Fd_(j_index, k) + qp_Fi_(j_index, k) - qp_Fg_(j_index, k));
            }
        }
        for (auto k = 0U; k < 6U; ++k) {
            residual_vector_terms_(i_elem, i_index, k) = local_residual[k] - node_FX_(i_index, k);
        }
    }
};

struct IntegrateResidualVector {
    Kokkos::View<size_t*>::const_type num_nodes_per_element;
    Kokkos::View<size_t*>::const_type num_qps_per_element;
    Kokkos::View<double**>::const_type qp_weight_;
    Kokkos::View<double**>::const_type qp_jacobian_;
    Kokkos::View<double***>::const_type shape_interp_;
    Kokkos::View<double***>::const_type shape_deriv_;
    Kokkos::View<double** [6]>::const_type node_FX_;
    Kokkos::View<double** [6]>::const_type qp_Fc_;
    Kokkos::View<double** [6]>::const_type qp_Fd_;
    Kokkos::View<double** [6]>::const_type qp_Fi_;
    Kokkos::View<double** [6]>::const_type qp_Fg_;
    Kokkos::View<double** [6]> residual_vector_terms_;

    KOKKOS_FUNCTION void operator()(const Kokkos::TeamPolicy<>::member_type& member) const {
        const auto i_elem = static_cast<size_t>(member.league_rank());
        const auto num_nodes = num_nodes_per_element(i_elem);
        const auto num_qps = num_qps_per_element(i_elem);

        const auto shape_interp = Kokkos::View<double**>(member.team_scratch(1), num_nodes, num_qps);
        const auto shape_deriv = Kokkos::View<double**>(member.team_scratch(1), num_nodes, num_qps);

        const auto qp_weight = Kokkos::View<double*>(member.team_scratch(1), num_qps);
        const auto qp_jacobian = Kokkos::View<double*>(member.team_scratch(1), num_qps);

        const auto node_FX = Kokkos::View<double* [6]>(member.team_scratch(1), num_nodes);
        const auto qp_Fc = Kokkos::View<double* [6]>(member.team_scratch(1), num_qps);
        const auto qp_Fd = Kokkos::View<double* [6]>(member.team_scratch(1), num_qps);
        const auto qp_Fi = Kokkos::View<double* [6]>(member.team_scratch(1), num_qps);
        const auto qp_Fg = Kokkos::View<double* [6]>(member.team_scratch(1), num_qps);

        Kokkos::parallel_for(Kokkos::TeamThreadRange(member, num_qps), [&](size_t k) {
            for (auto i = 0U; i < num_nodes; ++i) {
                shape_interp(i, k) = shape_interp_(i_elem, i, k);
                shape_deriv(i, k) = shape_deriv_(i_elem, i, k);
            }
            qp_weight(k) = qp_weight_(i_elem, k);
            qp_jacobian(k) = qp_jacobian_(i_elem, k);
            for (auto i = 0U; i < 6U; ++i) {
                qp_Fc(k, i) = qp_Fc_(i_elem, k, i);
                qp_Fd(k, i) = qp_Fd_(i_elem, k, i);
                qp_Fi(k, i) = qp_Fi_(i_elem, k, i);
                qp_Fg(k, i) = qp_Fg_(i_elem, k, i);
            }
        });
        Kokkos::parallel_for(Kokkos::TeamThreadRange(member, num_nodes), [&](size_t k) {
            for (auto i = 0U; i < 6U; ++i) {
                node_FX(k, i) = node_FX_(i_elem, k, i);
            }
        });
        member.team_barrier();

        const auto node_range = Kokkos::TeamThreadRange(member, num_nodes);
        const auto element_integrator = IntegrateResidualVectorElement{
            i_elem,  num_qps, qp_weight, qp_jacobian, shape_interp, shape_deriv,
            node_FX, qp_Fc,   qp_Fd,     qp_Fi,       qp_Fg,        residual_vector_terms_};
        Kokkos::parallel_for(node_range, element_integrator);
    }
};

}  // namespace openturbine
