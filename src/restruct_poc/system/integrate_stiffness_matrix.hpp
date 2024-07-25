#pragma once

#include <Kokkos_Core.hpp>

#include "src/restruct_poc/beams/beams.hpp"
#include "src/restruct_poc/types.hpp"

namespace openturbine {

struct IntegrateStiffnessMatrix {
    Kokkos::View<Beams::ElemIndices*>::const_type elem_indices;
    View_N::const_type qp_weight_;
    View_N::const_type qp_jacobian_;
    Kokkos::View<const double**, Kokkos::MemoryTraits<Kokkos::RandomAccess>> shape_interp_;
    Kokkos::View<const double**, Kokkos::MemoryTraits<Kokkos::RandomAccess>> shape_deriv_;
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
        Kokkos::parallel_for(
            Kokkos::TeamThreadMDRange(member, idx.num_nodes, idx.num_nodes),
            [&](size_t i_index, size_t j_index) {
                auto local_M_data = Kokkos::Array<double, 36>{};
                auto local_M = Kokkos::View<double[6][6], Kokkos::MemoryTraits<Kokkos::Unmanaged>>(
                    local_M_data.data()
                );
                const auto i = i_index + idx.node_range.first;
                const auto j = j_index + idx.node_range.first;
                for (auto k = 0u; k < idx.num_qps; ++k) {
                    const auto k_qp = idx.qp_range.first + k;
                    const auto w = qp_weight_(k_qp);
                    const auto jacobian = qp_jacobian_(k_qp);
                    const auto phi_i = shape_interp_(i, k);
                    const auto phi_j = shape_interp_(j, k);
                    const auto phi_prime_i = shape_deriv_(i, k);
                    const auto phi_prime_j = shape_deriv_(j, k);
                    const auto K = w * phi_i * phi_j * jacobian;
                    const auto P = w * (phi_i * phi_prime_j);
                    const auto Q = w * phi_i * phi_j * jacobian;
                    const auto C = w * (phi_prime_i * phi_prime_j / jacobian);
                    const auto O = w * (phi_prime_i * phi_j);
                    for (auto m = 0u; m < 6u; ++m) {
                        for (auto n = 0u; n < 6u; ++n) {
                            local_M(m, n) += K * qp_Kuu_(k_qp, m, n) + P * qp_Puu_(k_qp, m, n) +
                                             Q * qp_Quu_(k_qp, m, n) + C * qp_Cuu_(k_qp, m, n) +
                                             O * qp_Ouu_(k_qp, m, n);
                        }
                    }
                }
                for (auto m = 0u; m < 6u; ++m) {
                    for (auto n = 0u; n < 6u; ++n) {
                        gbl_M_(i_elem, i_index * 6u + m, j_index * 6u + n) = local_M(m, n);
                    }
                }
            }
        );
    }
};

}  // namespace openturbine
