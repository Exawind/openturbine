#pragma once

#include <Kokkos_Core.hpp>

#include "src/restruct_poc/types.hpp"

namespace openturbine {

struct InterpolateQPState_u {
    int first_qp;
    int first_node;
    int num_nodes;
    View_NxN::const_type shape_interp;
    View_Nx7::const_type node_u;
    View_Nx3 qp_u;

    KOKKOS_FUNCTION
    void operator()(int j_index) const {
        const auto j = first_qp + j_index;
        auto local_total = Kokkos::Array<double, 3>{};
        for (int i_index = 0; i_index < num_nodes; ++i_index) {
            const auto i = first_node + i_index;
            const auto phi = shape_interp(i, j_index);
            for (int k = 0; k < 3; ++k) {
                local_total[k] += node_u(i, k) * phi;
            }
        }
        for (int k = 0; k < 3; ++k) {
            qp_u(j, k) = local_total[k];
        }
    }
};

struct InterpolateQPState_uprime {
    int first_qp;
    int first_node;
    int num_nodes;
    View_NxN::const_type shape_deriv;
    View_N::const_type qp_jacobian;
    View_Nx7::const_type node_u;
    View_Nx3 qp_uprime;

    KOKKOS_FUNCTION
    void operator()(int j_index) const {
        const auto j = first_qp + j_index;
        const auto jacobian = qp_jacobian(j);
        auto local_total = Kokkos::Array<double, 3>{};
        for (int i_index = 0; i_index < num_nodes; ++i_index) {
            const auto i = first_node + i_index;
            const auto dphi = shape_deriv(i, j_index);
            for (int k = 0; k < 3; ++k) {
                local_total[k] += node_u(i, k) * dphi / jacobian;
            }
        }
        for (int k = 0; k < 3; ++k) {
            qp_uprime(j, k) = local_total[k];
        }
    }
};

struct InterpolateQPState_r {
    int first_qp;
    int first_node;
    int num_nodes;
    View_NxN::const_type shape_interp;
    View_Nx7::const_type node_u;
    View_Nx4 qp_r;

    KOKKOS_FUNCTION
    void operator()(int j_index) const {
        const auto j = first_qp + j_index;
        auto local_total = Kokkos::Array<double, 4>{};
        for (int i_index = 0; i_index < num_nodes; ++i_index) {
            const auto i = first_node + i_index;
            const auto phi = shape_interp(i, j_index);
            for (int k = 0; k < 4; ++k) {
                local_total[k] += node_u(i, k + 3) * phi;
            }
        }
        const auto length = Kokkos::sqrt(
            local_total[0] * local_total[0] + local_total[1] * local_total[1] +
            local_total[2] * local_total[2] + local_total[3] * local_total[3]
        );
        static constexpr auto length_zero_result = Kokkos::Array<double, 4>{1., 0., 0., 0.};
        if (length == 0.) {
            local_total = length_zero_result;
        }
        for (int k = 0; k < 4; ++k) {
            qp_r(j, k) = local_total[k];
        }
    }
};

struct InterpolateQPState_rprime {
    int first_qp;
    int first_node;
    int num_nodes;
    View_NxN::const_type shape_deriv;
    View_N::const_type qp_jacobian;
    View_Nx7::const_type node_u;
    View_Nx4 qp_rprime;

    KOKKOS_FUNCTION
    void operator()(int j_index) const {
        const auto j = first_qp + j_index;
        const auto jacobian = qp_jacobian(j);
        auto local_total = Kokkos::Array<double, 4>{};
        for (int i_index = 0; i_index < num_nodes; ++i_index) {
            const auto i = first_node + i_index;
            const auto dphi = shape_deriv(i, j_index);
            for (int k = 0; k < 4; ++k) {
                local_total[k] += node_u(i, k + 3) * dphi / jacobian;
            }
        }
        for (int k = 0; k < 4; ++k) {
            qp_rprime(j, k) = local_total[k];
        }
    }
};

}  // namespace openturbine
