#pragma once

#include <Kokkos_Core.hpp>

#include "src/restruct_poc/beams/beams.hpp"
#include "src/restruct_poc/types.hpp"

namespace openturbine {

struct IntegrateStiffnessMatrix {
    Kokkos::View<Beams::ElemIndices*>::const_type elem_indices;  // Element indices
    View_N::const_type qp_weight_;                               //
    View_N::const_type qp_jacobian_;                             // Jacobians
    View_NxN::const_type shape_interp_;                          // Num Nodes x Num Quadrature points
    View_NxN::const_type shape_deriv_;                           // Num Nodes x Num Quadrature points
    View_Nx6x6::const_type qp_Kuu_;
    View_Nx6x6::const_type qp_Puu_;                              //
    View_Nx6x6::const_type qp_Cuu_;                              //
    View_Nx6x6::const_type qp_Ouu_;                              //
    View_Nx6x6::const_type qp_Quu_;                              //
    Kokkos::View<double***> gbl_M_;

    KOKKOS_FUNCTION
    void operator()(Kokkos::TeamPolicy<>::member_type member) const {
        const auto i_elem = member.league_rank();
        const auto idx = elem_indices(i_elem);
        Kokkos::parallel_for(
            Kokkos::TeamThreadMDRange(member, idx.num_nodes, idx.num_nodes),
            [=](int i_index, int j_index) {
                const auto i = i_index + idx.node_range.first;
                const auto j = j_index + idx.node_range.first;
                auto local_M_data = Kokkos::Array<double, 36>{};
                auto local_M = Kokkos::View<double[6][6], Kokkos::MemoryTraits<Kokkos::Unmanaged>>(
                    local_M_data.data()
                );
                for (int k = 0; k < idx.num_qps; ++k) {
                    const auto k_qp = idx.qp_range.first + k;
                    const auto w = qp_weight_(k_qp);
                    const auto jacobian = qp_jacobian_(k_qp);
                    const auto phi_i = shape_interp_(i, k);
                    const auto phi_j = shape_interp_(j, k);
                    const auto phi_prime_i = shape_deriv_(i, k);
                    const auto phi_prime_j = shape_deriv_(j, k);
                    const auto coeff_K = w * (phi_i * phi_j * jacobian);
                    const auto coeff_P = w * (phi_i * phi_prime_j);
                    const auto coeff_Q = w * (phi_i * phi_j * jacobian);
                    const auto coeff_C = w * (phi_prime_i * phi_prime_j / jacobian);
                    const auto coeff_O = w * (phi_prime_i * phi_j);
                    Kokkos::parallel_for(
                        Kokkos::ThreadVectorMDRange(member, 6, 6),
                        [=](int m, int n) {
                            local_M(m, n) += coeff_K * qp_Kuu_(k_qp, m, n) +
                                coeff_P * qp_Puu_(k_qp, m, n) + coeff_Q * qp_Quu_(k_qp, m, n) +
                                coeff_C * qp_Cuu_(k_qp, m, n) + coeff_O * qp_Ouu_(k_qp, m, n);
                        }
                    );
                }

                Kokkos::parallel_for(Kokkos::ThreadVectorMDRange(member, 6, 6), [=](int m, int n) {
                    gbl_M_(i_elem, i_index*6 + m, j_index*6 + n) = local_M(m, n);
                });
            }
        );
    }
};

}  // namespace openturbine
