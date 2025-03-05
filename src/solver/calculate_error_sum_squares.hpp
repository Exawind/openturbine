#pragma once

#include <Kokkos_Core.hpp>

namespace openturbine {

struct CalculateSystemErrorSumSquares {
    using value_type = double;
    double atol;
    double rtol;
    double h;
    Kokkos::View<FreedomSignature*>::const_type node_freedom_allocation_table;
    Kokkos::View<size_t*>::const_type node_freedom_map_table;
    Kokkos::View<double* [6]>::const_type q_delta;
    Kokkos::View<double* [1], Kokkos::LayoutLeft>::const_type x;

    KOKKOS_INLINE_FUNCTION
    void operator()(const size_t i_node, double& err) const {
        const auto n_node_dofs = count_active_dofs(node_freedom_allocation_table(i_node));
        const auto node_first_dof = node_freedom_map_table(i_node);
        for (auto j = 0U; j < n_node_dofs; ++j) {
            const auto pi = x(node_first_dof + j, 0);
            const auto xi = q_delta(i_node, j) * h;
            err += Kokkos::pow(pi / (atol + rtol * Kokkos::abs(xi)), 2.);
        }
    }
};

struct CalculateConstraintsErrorSumSquares {
    using value_type = double;
    double atol;
    double rtol;
    size_t num_system_dofs;
    Kokkos::View<Kokkos::pair<size_t, size_t>*>::const_type row_range;
    Kokkos::View<double* [6]>::const_type lambda;
    Kokkos::View<double* [1], Kokkos::LayoutLeft>::const_type x;

    KOKKOS_INLINE_FUNCTION
    void operator()(const size_t i_constraint, double& err) const {
        const auto first_index = row_range(i_constraint).first;
        const auto max_index = row_range(i_constraint).second;
        for (auto row = first_index; row < max_index; ++row) {
            const auto pi = x(num_system_dofs + row, 0);
            const auto xi = lambda(i_constraint, row - first_index);
            err += Kokkos::pow(pi / (atol + rtol * Kokkos::abs(xi)), 2.);
        }
    }
};

}  // namespace openturbine
