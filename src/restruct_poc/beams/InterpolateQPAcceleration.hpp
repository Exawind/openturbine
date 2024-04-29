#pragma once

#include <Kokkos_Core.hpp>

#include "Beams.hpp"
#include "src/restruct_poc/types.hpp"

namespace openturbine {

struct InterpolateQPAcceleration_Translation {
    Kokkos::View<Beams::ElemIndices*>::const_type elem_indices;  // Element indices
    View_NxN::const_type shape_interp_;                          // Num Nodes x Num Quadrature points
    View_Nx6::const_type node_u_ddot_;  // Node translation & angular velocity
    View_Nx3 qp_u_ddot_;                // qp translation velocity

    KOKKOS_FUNCTION
    void operator()(const int i_elem) const {
        auto idx = elem_indices(i_elem);
        for (int j_index = 0; j_index < idx.num_qps; ++j_index) {
            const auto j = idx.qp_range.first + j_index;
            auto local_total = Kokkos::Array<double, 3>{};
            for (int i_index = 0; i_index < idx.num_nodes; ++i_index) {
                const auto i = idx.node_range.first + i_index;
                const auto phi = shape_interp_(i, j_index);
                for (int k = 0; k < 3; ++k) {
                    local_total[k] += node_u_ddot_(i, k) * phi;
                }
            }
            for (int k = 0; k < 3; ++k) {
                qp_u_ddot_(j, k) = local_total[k];
            }
        }
    }

    KOKKOS_FUNCTION
    void operator()(const int i_elem, const int j_index) const {
        auto idx = elem_indices(i_elem);
        if (j_index >= idx.num_qps) {
            return;
        }

        const auto j = idx.qp_range.first + j_index;
        auto local_total = Kokkos::Array<double, 3>{};
        for (int i_index = 0; i_index < idx.num_nodes; ++i_index) {
            const auto i = idx.node_range.first + i_index;
            const auto phi = shape_interp_(i, j_index);
            for (int k = 0; k < 3; ++k) {
                local_total[k] += node_u_ddot_(i, k) * phi;
            }
        }
        for (int k = 0; k < 3; ++k) {
            qp_u_ddot_(j, k) = local_total[k];
        }
    }
};

struct InterpolateQPAcceleration_Angular {
    Kokkos::View<Beams::ElemIndices*>::const_type elem_indices;  // Element indices
    View_NxN::const_type shape_interp_;                          // Num Nodes x Num Quadrature points
    View_Nx6::const_type node_u_ddot_;  // Node translation & angular velocity
    View_Nx3 qp_omega_dot_;             // qp angular velocity

    KOKKOS_FUNCTION
    void operator()(const int i_elem) const {
        auto idx = elem_indices(i_elem);
        for (int j_index = 0; j_index < idx.num_qps; ++j_index) {
            const auto j = idx.qp_range.first + j_index;
            auto local_total = Kokkos::Array<double, 3>{};
            for (int i_index = 0; i_index < idx.num_nodes; ++i_index) {
                const auto i = idx.node_range.first + i_index;
                const auto phi = shape_interp_(i, j_index);
                for (int k = 0; k < 3; ++k) {
                    local_total[k] += node_u_ddot_(i, k + 3) * phi;
                }
            }
            for (int k = 0; k < 3; ++k) {
                qp_omega_dot_(j, k) = local_total[k];
            }
        }
    }

    KOKKOS_FUNCTION
    void operator()(const int i_elem, const int j_index) const {
        auto idx = elem_indices(i_elem);
        if (j_index >= idx.num_qps) {
            return;
        }

        const auto j = idx.qp_range.first + j_index;
        auto local_total = Kokkos::Array<double, 3>{};
        for (int i_index = 0; i_index < idx.num_nodes; ++i_index) {
            const auto i = idx.node_range.first + i_index;
            const auto phi = shape_interp_(i, j_index);
            for (int k = 0; k < 3; ++k) {
                local_total[k] += node_u_ddot_(i, k + 3) * phi;
            }
        }
        for (int k = 0; k < 3; ++k) {
            qp_omega_dot_(j, k) = local_total[k];
        }
    }
};

}  // namespace openturbine
