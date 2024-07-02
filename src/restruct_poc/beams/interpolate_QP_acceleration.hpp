#pragma once

#include <Kokkos_Core.hpp>

#include "beams.hpp"

#include "src/restruct_poc/types.hpp"

namespace openturbine {

struct InterpolateQPAcceleration_Translational {
    int first_qp;
    int first_node;
    int num_nodes;
    View_NxN::const_type shape_interp;
    View_Nx6::const_type node_u_ddot; 
    View_Nx3 qp_u_ddot;

    KOKKOS_FUNCTION
    void operator()(int j_index) const {
        const auto j = first_qp + j_index;
        auto local_total = Kokkos::Array<double, 3>{};
        for (int i_index = 0; i_index < num_nodes; ++i_index) {
            const auto i = first_node + i_index;
            const auto phi = shape_interp(i, j_index);
            for (int k = 0; k < 3; ++k) {
                local_total[k] += node_u_ddot(i, k) * phi;
            }
        }
        for (int k = 0; k < 3; ++k) {
            qp_u_ddot(j, k) = local_total[k];
        }
    }
};

struct InterpolateQPAcceleration_Angular {
    int first_qp;
    int first_node;
    int num_nodes;
    View_NxN::const_type shape_interp;
    View_Nx6::const_type node_u_ddot; 
    View_Nx3 qp_omega_dot;

    KOKKOS_FUNCTION
    void operator()(int j_index) const {
        const auto j = first_qp + j_index;
        auto local_total = Kokkos::Array<double, 3>{};
        for (int i_index = 0; i_index < num_nodes; ++i_index) {
            const auto i = first_node + i_index;
            const auto phi = shape_interp(i, j_index);
            for (int k = 0; k < 3; ++k) {
                local_total[k] += node_u_ddot(i, k + 3) * phi;
            }
        }
        for (int k = 0; k < 3; ++k) {
            qp_omega_dot(j, k) = local_total[k];
        }
    }
};
struct InterpolateQPAcceleration {
    Kokkos::View<Beams::ElemIndices*>::const_type elem_indices; 
    View_NxN::const_type shape_interp;
    View_Nx6::const_type node_u_ddot; 
    View_Nx3 qp_u_ddot;
    View_Nx3 qp_omega_dot;

    KOKKOS_FUNCTION
    void operator()(Kokkos::TeamPolicy<>::member_type member) const {
        const auto i_elem = member.league_rank();
        const auto idx = elem_indices(i_elem);
        const auto first_qp = idx.qp_range.first;
        const auto first_node = idx.node_range.first;
        const auto num_nodes = idx.num_nodes;
        Kokkos::parallel_for(Kokkos::TeamThreadRange(member, idx.num_qps), InterpolateQPAcceleration_Translational{first_qp, first_node, num_nodes, shape_interp, node_u_ddot, qp_u_ddot});
        Kokkos::parallel_for(Kokkos::TeamThreadRange(member, idx.num_qps), InterpolateQPAcceleration_Angular{first_qp, first_node, num_nodes, shape_interp, node_u_ddot, qp_omega_dot});
    }
};

}  // namespace openturbine
