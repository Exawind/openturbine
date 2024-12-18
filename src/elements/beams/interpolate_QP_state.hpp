#pragma once

#include <Kokkos_Core.hpp>

#include "src/math/quaternion_operations.hpp"

namespace openturbine {

/**
 * @brief Interpolates the displacement (u) part of the state at a quadrature point
 */
struct InterpolateQPState_u {
    size_t i_elem;
    size_t num_nodes;
    Kokkos::View<double***>::const_type shape_interp;
    Kokkos::View<double** [7]>::const_type node_u;
    Kokkos::View<double** [3]> qp_u;

    KOKKOS_FUNCTION
    void operator()(size_t j_index) const {
        auto local_total = Kokkos::Array<double, 3>{};
        for (auto i_index = 0U; i_index < num_nodes; ++i_index) {
            const auto phi = shape_interp(i_elem, i_index, j_index);
            for (auto k = 0U; k < 3U; ++k) {
                local_total[k] += node_u(i_elem, i_index, k) * phi;
            }
        }
        for (auto k = 0U; k < 3U; ++k) {
            qp_u(i_elem, j_index, k) = local_total[k];
        }
    }
};

/**
 * @brief Interpolates the displacement derivative (u') part of the state at a quadrature point
 */
struct InterpolateQPState_uprime {
    size_t i_elem;
    size_t num_nodes;
    Kokkos::View<double***>::const_type shape_deriv;
    Kokkos::View<double**>::const_type qp_jacobian;
    Kokkos::View<double** [7]>::const_type node_u;
    Kokkos::View<double** [3]> qp_uprime;

    KOKKOS_FUNCTION
    void operator()(size_t j_index) const {
        const auto jacobian = qp_jacobian(i_elem, j_index);
        auto local_total = Kokkos::Array<double, 3>{};
        for (auto i_index = 0U; i_index < num_nodes; ++i_index) {
            const auto dphi = shape_deriv(i_elem, i_index, j_index);
            for (auto k = 0U; k < 3U; ++k) {
                local_total[k] += node_u(i_elem, i_index, k) * dphi / jacobian;
            }
        }
        for (auto k = 0U; k < 3U; ++k) {
            qp_uprime(i_elem, j_index, k) = local_total[k];
        }
    }
};

/**
 * @brief Interpolates the rotation (r) part of the state at a quadrature point
 */
struct InterpolateQPState_r {
    size_t i_elem;
    size_t num_nodes;
    Kokkos::View<double***>::const_type shape_interp;
    Kokkos::View<double** [7]>::const_type node_u;
    Kokkos::View<double** [4]> qp_r;

    KOKKOS_FUNCTION
    void operator()(size_t j_index) const {
        auto local_total = Kokkos::Array<double, 4>{};
        for (auto i_index = 0U; i_index < num_nodes; ++i_index) {
            const auto phi = shape_interp(i_elem, i_index, j_index);
            for (auto k = 0U; k < 4U; ++k) {
                local_total[k] += node_u(i_elem, i_index, k + 3) * phi;
            }
        }

        for (auto k = 0U; k < 4U; ++k) {
            qp_r(i_elem, j_index, k) = NormalizeQuaternion(local_total)[k];
        }
    }
};

/**
 * @brief Interpolates the rotation derivative (r') part of the state at a quadrature point
 */
struct InterpolateQPState_rprime {
    size_t i_elem;
    size_t num_nodes;
    Kokkos::View<double***>::const_type shape_deriv;
    Kokkos::View<double**>::const_type qp_jacobian;
    Kokkos::View<double** [7]>::const_type node_u;
    Kokkos::View<double** [4]> qp_rprime;

    KOKKOS_FUNCTION
    void operator()(size_t j_index) const {
        const auto jacobian = qp_jacobian(i_elem, j_index);
        auto local_total = Kokkos::Array<double, 4>{};
        for (auto i_index = 0U; i_index < num_nodes; ++i_index) {
            const auto dphi = shape_deriv(i_elem, i_index, j_index);
            for (auto k = 0U; k < 4U; ++k) {
                local_total[k] += node_u(i_elem, i_index, k + 3) * dphi / jacobian;
            }
        }
        for (auto k = 0U; k < 4U; ++k) {
            qp_rprime(i_elem, j_index, k) = local_total[k];
        }
    }
};

}  // namespace openturbine
