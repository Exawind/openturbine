#pragma once

#include <Kokkos_Core.hpp>
#include <Kokkos_SIMD.hpp>

#include "src/restruct_poc/beams/beams.hpp"
#include "src/restruct_poc/types.hpp"

namespace openturbine {
struct IntegrateStiffnessMatrixElement {
    size_t i_elem;
    size_t num_nodes;
    size_t num_qps;
    Kokkos::View<double*>::const_type qp_weight_;
    Kokkos::View<double*>::const_type qp_jacobian_;
    Kokkos::View<double**, Kokkos::LayoutLeft>::const_type shape_interp_;
    Kokkos::View<double**, Kokkos::LayoutLeft>::const_type shape_deriv_;
    Kokkos::View<double* [6][6]>::const_type qp_Kuu_;
    Kokkos::View<double* [6][6]>::const_type qp_Puu_;
    Kokkos::View<double* [6][6]>::const_type qp_Cuu_;
    Kokkos::View<double* [6][6]>::const_type qp_Ouu_;
    Kokkos::View<double* [6][6]>::const_type qp_Quu_;
    Kokkos::View<double***> gbl_M_;

    KOKKOS_FUNCTION
    void operator()(size_t i_index) const {
        using simd_type = Kokkos::Experimental::native_simd<double>;
        using mask_type = Kokkos::Experimental::native_simd_mask<double>;
        using tag_type = Kokkos::Experimental::element_aligned_tag;
        constexpr auto width = simd_type::size();
        for (auto j_index = 0U; j_index < num_nodes; j_index += width) {
            auto mask = mask_type([j_index, num_nodes = this->num_nodes](size_t lane) {
                return j_index + lane < num_nodes;
            });
            auto local_M_data = Kokkos::Array<simd_type, 36>{};
            const auto local_M = Kokkos::View<simd_type[6][6]>(local_M_data.data());
            for (auto k = 0U; k < num_qps; ++k) {
                const auto w = qp_weight_(k);
                const auto jacobian = qp_jacobian_(k);
                const auto phi_i = shape_interp_(i_index, k);
                auto phi_j = simd_type{};
                Kokkos::Experimental::where(mask, phi_j)
                    .copy_from(&shape_interp_(j_index, k), tag_type());
                const auto phi_prime_i = shape_deriv_(i_index, k);
                auto phi_prime_j = simd_type{};
                Kokkos::Experimental::where(mask, phi_prime_j)
                    .copy_from(&shape_deriv_(j_index, k), tag_type());
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
                for (auto lane = 0U; lane < width && mask[lane]; ++lane) {
                    for (auto n = 0U; n < 6U; ++n) {
                        gbl_M_(i_elem, i_index * 6 + m, (j_index + lane) * 6 + n) =
                            local_M(m, n)[lane];
                    }
                }
            }
        }
    }
};

struct IntegrateStiffnessMatrix {
    Kokkos::View<size_t*>::const_type num_nodes_per_element;
    Kokkos::View<size_t*>::const_type num_qps_per_element;
    Kokkos::View<double**>::const_type qp_weight_;
    Kokkos::View<double**>::const_type qp_jacobian_;
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
        const auto i_elem = static_cast<size_t>(member.league_rank());
        const auto num_nodes = num_nodes_per_element(i_elem);
        const auto num_qps = num_qps_per_element(i_elem);

        const auto shape_interp =
            Kokkos::View<double**, Kokkos::LayoutLeft>(member.team_scratch(1), num_nodes, num_qps);
        const auto shape_deriv =
            Kokkos::View<double**, Kokkos::LayoutLeft>(member.team_scratch(1), num_nodes, num_qps);

        const auto qp_weight = Kokkos::View<double*>(member.team_scratch(1), num_qps);
        const auto qp_jacobian = Kokkos::View<double*>(member.team_scratch(1), num_qps);

        const auto qp_Kuu = Kokkos::View<double* [6][6]>(member.team_scratch(1), num_qps);
        const auto qp_Puu = Kokkos::View<double* [6][6]>(member.team_scratch(1), num_qps);
        const auto qp_Cuu = Kokkos::View<double* [6][6]>(member.team_scratch(1), num_qps);
        const auto qp_Ouu = Kokkos::View<double* [6][6]>(member.team_scratch(1), num_qps);
        const auto qp_Quu = Kokkos::View<double* [6][6]>(member.team_scratch(1), num_qps);

        Kokkos::parallel_for(Kokkos::TeamThreadRange(member, num_qps), [&](size_t k) {
            for (auto i = 0U; i < num_nodes; ++i) {
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

        const auto node_range = Kokkos::TeamThreadRange(member, num_nodes);
        const auto element_integrator = IntegrateStiffnessMatrixElement{
            i_elem, num_nodes, num_qps, qp_weight, qp_jacobian, shape_interp, shape_deriv,
            qp_Kuu, qp_Puu,    qp_Cuu,  qp_Ouu,    qp_Quu,      gbl_M_};
        Kokkos::parallel_for(node_range, element_integrator);
    }
};

}  // namespace openturbine
