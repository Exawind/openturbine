#pragma once

#include <Kokkos_Core.hpp>

#include "dof_management/freedom_signature.hpp"

namespace openturbine {

struct ComputeTNumNonZeroReducer {
    Kokkos::View<FreedomSignature*>::const_type node_freedom_allocation_table;

    KOKKOS_FUNCTION
    void operator()(size_t i, size_t& update) const {
        const auto num_node_dof = count_active_dofs(node_freedom_allocation_table(i));
        const auto num_diagonal_non_zero = num_node_dof * num_node_dof;
        update += num_diagonal_non_zero;
    }
};

[[nodiscard]] inline size_t ComputeTNumNonZero(
    const Kokkos::View<FreedomSignature*>::const_type& node_freedom_allocation_table
) {
    auto T_num_non_zero = 0UL;
    Kokkos::parallel_reduce(
        "ComputeTNumNonZero", node_freedom_allocation_table.extent(0),
        ComputeTNumNonZeroReducer{node_freedom_allocation_table}, T_num_non_zero
    );

    return T_num_non_zero;
}
}  // namespace openturbine
