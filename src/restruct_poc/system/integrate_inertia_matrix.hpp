#pragma once

#include <Kokkos_Core.hpp>

#include "src/restruct_poc/beams/beams.hpp"
#include "src/restruct_poc/types.hpp"

namespace openturbine {

struct IntegrateInertiaMatrixElement {
    size_t i_elem;
    size_t num_qps;
    Kokkos::View<double*>::const_type qp_weight_;
    Kokkos::View<double*>::const_type qp_jacobian_;
    Kokkos::View<double**>::const_type shape_interp_;
    Kokkos::View<double* [6][6]>::const_type qp_Muu_;
    Kokkos::View<double* [6][6]>::const_type qp_Guu_;
    double beta_prime_;
    double gamma_prime_;
    Kokkos::View<double*** [6][6]> gbl_M_;

    KOKKOS_FUNCTION
    void operator()(size_t i_index, size_t j_index) const {
        auto local_M_data = Kokkos::Array<double, 36>{};
        const auto local_M = Kokkos::View<double[6][6]>(local_M_data.data());
        for (auto k = 0U; k < num_qps; ++k) {
            const auto w = qp_weight_(k);
            const auto jacobian = qp_jacobian_(k);
            const auto phi_i = shape_interp_(i_index, k);
            const auto phi_j = shape_interp_(j_index, k);
            const auto coeff = w * phi_i * phi_j * jacobian;
            for (auto m = 0U; m < 6U; ++m) {
                for (auto n = 0U; n < 6U; ++n) {
                    local_M(m, n) +=
                        coeff * (beta_prime_ * qp_Muu_(k, m, n) + gamma_prime_ * qp_Guu_(k, m, n));
                }
            }
        }
        for (auto m = 0U; m < 6U; ++m) {
            for (auto n = 0U; n < 6U; ++n) {
                gbl_M_(i_elem, i_index, j_index, m, n) = local_M(m, n);
            }
        }
    }
};

struct IntegrateInertiaMatrix {
    Kokkos::View<size_t*>::const_type num_nodes_per_element;
    Kokkos::View<size_t*>::const_type num_qps_per_element;
    Kokkos::View<double**>::const_type qp_weight_;
    Kokkos::View<double**>::const_type qp_jacobian_;
    Kokkos::View<double***>::const_type shape_interp_;
    Kokkos::View<double** [6][6]>::const_type qp_Muu_;
    Kokkos::View<double** [6][6]>::const_type qp_Guu_;
    double beta_prime_;
    double gamma_prime_;
    Kokkos::View<double*** [6][6]> gbl_M_;

    KOKKOS_FUNCTION
    void operator()(const Kokkos::TeamPolicy<>::member_type& member) const {
        const auto i_elem = static_cast<size_t>(member.league_rank());
        const auto num_nodes = num_nodes_per_element(i_elem);
        const auto num_qps = num_qps_per_element(i_elem);
        const auto shape_interp = Kokkos::View<double**>(member.team_scratch(1), num_nodes, num_qps);

        const auto qp_weight = Kokkos::View<double*>(member.team_scratch(1), num_qps);
        const auto qp_jacobian = Kokkos::View<double*>(member.team_scratch(1), num_qps);

        const auto qp_Muu = Kokkos::View<double* [6][6]>(member.team_scratch(1), num_qps);
        const auto qp_Guu = Kokkos::View<double* [6][6]>(member.team_scratch(1), num_qps);

        Kokkos::parallel_for(Kokkos::TeamThreadRange(member, num_qps), [&](size_t k) {
            for (auto i = 0U; i < num_nodes; ++i) {
                shape_interp(i, k) = shape_interp_(i_elem, i, k);
            }
            qp_weight(k) = qp_weight_(i_elem, k);
            qp_jacobian(k) = qp_jacobian_(i_elem, k);
            for (auto m = 0U; m < 6U; ++m) {
                for (auto n = 0U; n < 6U; ++n) {
                    qp_Muu(k, m, n) = qp_Muu_(i_elem, k, m, n);
                    qp_Guu(k, m, n) = qp_Guu_(i_elem, k, m, n);
                }
            }
        });
        member.team_barrier();

        const auto node_range = Kokkos::TeamThreadMDRange(member, num_nodes, num_nodes);
        const auto element_integrator =
            IntegrateInertiaMatrixElement{i_elem, num_qps, qp_weight,   qp_jacobian,  shape_interp,
                                          qp_Muu, qp_Guu,  beta_prime_, gamma_prime_, gbl_M_};
        Kokkos::parallel_for(node_range, element_integrator);
    }
};
}  // namespace openturbine
