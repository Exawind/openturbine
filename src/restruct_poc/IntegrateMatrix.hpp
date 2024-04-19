#pragma once

#include <Kokkos_Core.hpp>

#include "beams.hpp"
#include "types.hpp"

namespace openturbine {

struct IntegrateMatrix {
    Kokkos::View<Beams::ElemIndices*>::const_type elem_indices;  // Element indices
    Kokkos::View<int*>::const_type node_state_indices;           // Element indices
    View_N::const_type qp_weight_;                               //
    View_N::const_type qp_jacobian_;                             // Jacobians
    View_NxN::const_type shape_interp_;                          // Num Nodes x Num Quadrature points
    View_Nx6x6::const_type qp_M_;                                //
    View_NxN_atomic gbl_M_;                                      //

    KOKKOS_FUNCTION
    void operator()(const int i_elem) const {
        const auto idx = elem_indices[i_elem];

        for (int i = idx.node_range.first; i < idx.node_range.second; ++i) {      // Nodes
            for (int j = idx.node_range.first; j < idx.node_range.second; ++j) {  // Nodes
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
                    for (int m = 0; m < kLieAlgebraComponents; ++m) {
                        for (int n = 0; n < kLieAlgebraComponents; ++n) {
                            local_M(m, n) += coeff * qp_M_(k_qp, m, n);
                        }
                    }
                }
                const auto i_gbl_start = node_state_indices(i) * kLieAlgebraComponents;
                const auto j_gbl_start = node_state_indices(j) * kLieAlgebraComponents;
                for (int m = 0; m < kLieAlgebraComponents; ++m) {
                    for (int n = 0; n < kLieAlgebraComponents; ++n) {
                        gbl_M_(i_gbl_start + m, j_gbl_start + n) += local_M(m, n);
                    }
                }
            }
        }
    }

    KOKKOS_FUNCTION
    void operator()(Kokkos::TeamPolicy<>::member_type member) const {
        const auto idx = elem_indices(member.league_rank());
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
                    const auto coeff = w * phi_i * phi_j * jacobian;
                    Kokkos::parallel_for(
                        Kokkos::ThreadVectorMDRange(member, 6, 6),
                        [=](int m, int n) {
                            local_M(m, n) += coeff * qp_M_(k_qp, m, n);
                        }
                    );
                }
                const auto i_gbl_start = node_state_indices(i) * kLieAlgebraComponents;
                const auto j_gbl_start = node_state_indices(j) * kLieAlgebraComponents;
                Kokkos::parallel_for(Kokkos::ThreadVectorMDRange(member, 6, 6), [=](int m, int n) {
                    gbl_M_(i_gbl_start + m, j_gbl_start + n) += local_M(m, n);
                });
            }
        );
    }

    KOKKOS_FUNCTION
    void operator()(const int i_elem, const int i_index, const int j_index) const {
        const auto idx = elem_indices[i_elem];

        if (i_index >= idx.num_nodes || j_index >= idx.num_nodes) {
            return;
        }

        const auto i = i_index + idx.node_range.first;
        const auto j = j_index + idx.node_range.first;
        auto local_M_data = Kokkos::Array<double, 36>{};
        auto local_M =
            Kokkos::View<double[6][6], Kokkos::MemoryTraits<Kokkos::Unmanaged>>(local_M_data.data());
        for(int k = 0; k < idx.num_qps; ++k) {
            const auto k_qp = idx.qp_range.first + k;
            const auto w = qp_weight_(k_qp);
            const auto jacobian = qp_jacobian_(k_qp);
            const auto phi_i = shape_interp_(i, k);
            const auto phi_j = shape_interp_(j, k);
            const auto coeff = w * phi_i * phi_j * jacobian;
            for (int m = 0; m < kLieAlgebraComponents; ++m) {
                for (int n = 0; n < kLieAlgebraComponents; ++n) {
                    local_M(m, n) += coeff * qp_M_(k_qp, m, n);
                }
            }
        }
        const auto i_gbl_start = node_state_indices(i) * kLieAlgebraComponents;
        const auto j_gbl_start = node_state_indices(j) * kLieAlgebraComponents;
        for (int m = 0; m < kLieAlgebraComponents; ++m) {
            for (int n = 0; n < kLieAlgebraComponents; ++n) {
                gbl_M_(i_gbl_start + m, j_gbl_start + n) += local_M(m, n);
            }
        }
    }

    KOKKOS_FUNCTION
    void operator()(const int i_elem, const int i_index, const int j_index, const int k) const {
        const auto idx = elem_indices[i_elem];

        if (i_index >= idx.num_nodes || j_index >= idx.num_nodes || k >= idx.num_qps) {
            return;
        }

        const auto i = i_index + idx.node_range.first;
        const auto j = j_index + idx.node_range.first;
        auto local_M_data = Kokkos::Array<double, 36>{};
        auto local_M =
            Kokkos::View<double[6][6], Kokkos::MemoryTraits<Kokkos::Unmanaged>>(local_M_data.data());
        const auto k_qp = idx.qp_range.first + k;
        const auto w = qp_weight_(k_qp);
        const auto jacobian = qp_jacobian_(k_qp);
        const auto phi_i = shape_interp_(i, k);
        const auto phi_j = shape_interp_(j, k);
        const auto coeff = w * phi_i * phi_j * jacobian;
        for (int m = 0; m < kLieAlgebraComponents; ++m) {
            for (int n = 0; n < kLieAlgebraComponents; ++n) {
                local_M(m, n) += coeff * qp_M_(k_qp, m, n);
            }
        }
        const auto i_gbl_start = node_state_indices(i) * kLieAlgebraComponents;
        const auto j_gbl_start = node_state_indices(j) * kLieAlgebraComponents;
        for (int m = 0; m < kLieAlgebraComponents; ++m) {
            for (int n = 0; n < kLieAlgebraComponents; ++n) {
                gbl_M_(i_gbl_start + m, j_gbl_start + n) += local_M(m, n);
            }
        }
    }
};

}  // namespace openturbine
