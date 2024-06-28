#pragma once

#include <Kokkos_Core.hpp>

#include "src/restruct_poc/beams/beams.hpp"
#include "src/restruct_poc/types.hpp"

namespace openturbine {

struct IntegrateInertiaMatrix {
    Kokkos::View<Beams::ElemIndices*>::const_type elem_indices;  
    View_N::const_type qp_weight_;                               
    View_N::const_type qp_jacobian_;                      
    Kokkos::View<const double**, Kokkos::MemoryTraits<Kokkos::RandomAccess>> shape_interp_;
    View_Nx6x6::const_type qp_Muu_;                                
    View_Nx6x6::const_type qp_Guu_;
    double beta_prime_;
    double gamma_prime_;
    Kokkos::View<double***> gbl_M_;                                      

    KOKKOS_FUNCTION
    void operator()(Kokkos::TeamPolicy<>::member_type member) const {
        auto i_elem = member.league_rank();
        const auto idx = elem_indices(i_elem);
        Kokkos::parallel_for(Kokkos::TeamThreadMDRange(member, idx.num_nodes, idx.num_nodes), [&](int i_index, int j_index) {
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
                    const auto coeff = w * phi_i * phi_j * jacobian;
                    for(int m = 0; m < 6; ++m) {
                        for(int n = 0; n < 6; ++n) {
                            local_M(m, n) += beta_prime_ * coeff * qp_Muu_(k_qp, m, n) + gamma_prime_ * coeff * qp_Guu_(k_qp, m, n);
                        }
                    }
                }
                for(int m = 0; m < 6; ++m) {
                    for(int n = 0; n < 6; ++n) {
                        gbl_M_(i_elem, i_index*6 + m, j_index*6 + n) = local_M(m, n);
                    }
                }
            }
        );
    }
};
}  // namespace openturbine
