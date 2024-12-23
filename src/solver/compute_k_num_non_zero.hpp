#pragma once

#include <Kokkos_Core.hpp>

#include "src/dof_management/freedom_signature.hpp"

namespace openturbine {

struct ComputeKNumNonZero_OffDiagonal {
    Kokkos::View<size_t*>::const_type num_nodes_per_element;
    Kokkos::View<size_t**>::const_type node_state_indices;
    Kokkos::View<FreedomSignature*>::const_type node_freedom_allocation_table;

    KOKKOS_FUNCTION
    void operator()(size_t i, size_t& update) const {
        auto num_element_dof = 0UL;
        for (auto j = 0U; j < num_nodes_per_element(i); ++j) {
            const auto num_node_dof =
                count_active_dofs(node_freedom_allocation_table(node_state_indices(i, j)));
            num_element_dof += num_node_dof;
        }
        const auto num_element_non_zero = num_element_dof * num_element_dof;
        auto num_diagonal_non_zero = 0UL;
        for (auto j = 0U; j < num_nodes_per_element(i); ++j) {
            const auto num_node_dof =
                count_active_dofs(node_freedom_allocation_table(node_state_indices(i, j)));
            num_diagonal_non_zero += num_node_dof * num_node_dof;
        }
        update += num_element_non_zero - num_diagonal_non_zero;
    }
};

struct ComputeKNumNonZero_Diagonal {
    Kokkos::View<FreedomSignature*>::const_type node_freedom_allocation_table;

    KOKKOS_FUNCTION
    void operator()(size_t i, size_t& update) const {
        const auto num_node_dof = count_active_dofs(node_freedom_allocation_table(i));
        const auto num_diagonal_non_zero = num_node_dof * num_node_dof;
        update += num_diagonal_non_zero;
    }
};

/// Computes the number of non-zero entries in the stiffness matrix K for sparse storage
[[nodiscard]] inline size_t ComputeKNumNonZero(
    const Kokkos::View<size_t*>::const_type& num_nodes_per_element,
    const Kokkos::View<size_t**>::const_type& node_state_indices,
    const Kokkos::View<FreedomSignature*>::const_type& node_freedom_allocation_table
) {
    auto K_num_non_zero_off_diagonal = 0UL;
    Kokkos::parallel_reduce(
        "ComputeKNumNonZero_OffDiagonal", num_nodes_per_element.extent(0),
        ComputeKNumNonZero_OffDiagonal{
            num_nodes_per_element, node_state_indices, node_freedom_allocation_table
        },
        K_num_non_zero_off_diagonal
    );
    auto K_num_non_zero_diagonal = 0UL;
    Kokkos::parallel_reduce(
        "ComputeKNumNonZero_Diagonal", node_freedom_allocation_table.extent(0),
        ComputeKNumNonZero_Diagonal{node_freedom_allocation_table}, K_num_non_zero_diagonal
    );
    return K_num_non_zero_off_diagonal + K_num_non_zero_diagonal;
}

}  // namespace openturbine
