#pragma once

#include <Kokkos_Core.hpp>

namespace openturbine {

template <typename DeviceType>
struct CalculateSystemErrorSumSquares {
    using value_type = double;
    using member_type =
        typename Kokkos::TeamPolicy<typename DeviceType::execution_space>::member_type;
    double atol;
    double rtol;
    double h;
    typename Kokkos::View<size_t*, DeviceType>::const_type active_dofs;
    typename Kokkos::View<size_t*, DeviceType>::const_type node_freedom_map_table;
    typename Kokkos::View<double* [6], DeviceType>::const_type q_delta;
    typename Kokkos::View<double* [1], Kokkos::LayoutLeft, DeviceType>::const_type x;

    KOKKOS_INLINE_FUNCTION
    void operator()(const size_t i_node, double& err) const {
        const auto n_node_dofs = active_dofs(i_node);
        const auto node_first_dof = node_freedom_map_table(i_node);
        for (auto j = 0U; j < n_node_dofs; ++j) {
            const auto pi = x(node_first_dof + j, 0);
            const auto xi = q_delta(i_node, j) * h;
            const auto err_sqrt = pi / (atol + rtol * Kokkos::abs(xi));
            err += err_sqrt * err_sqrt;
        }
    }
};

template <typename DeviceType>
struct CalculateConstraintsErrorSumSquares {
    using value_type = double;
    double atol;
    double rtol;
    size_t num_system_dofs;
    typename Kokkos::View<Kokkos::pair<size_t, size_t>*, DeviceType>::const_type row_range;
    typename Kokkos::View<double* [6], DeviceType>::const_type lambda;
    typename Kokkos::View<double* [1], Kokkos::LayoutLeft, DeviceType>::const_type x;

    KOKKOS_INLINE_FUNCTION
    void operator()(const size_t i_constraint, double& err) const {
        const auto first_index = row_range(i_constraint).first;
        const auto max_index = row_range(i_constraint).second;
        for (auto row = first_index; row < max_index; ++row) {
            const auto pi = x(num_system_dofs + row, 0);
            const auto xi = lambda(i_constraint, row - first_index);
            const auto err_sqrt = pi / (atol + rtol * Kokkos::abs(xi));
            err += err_sqrt * err_sqrt;
        }
    }
};

}  // namespace openturbine
