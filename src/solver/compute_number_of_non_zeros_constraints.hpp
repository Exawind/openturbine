#pragma once

#include <Kokkos_Core.hpp>

#include "dof_management/freedom_signature.hpp"

namespace openturbine {

/// ComputeNumberOfNonZeros_Constraints calculates the total number of nonzero values in the
/// constraint gradient matrix based on the block layout of the constraints.
struct ComputeNumberOfNonZeros_Constraints {
    Kokkos::View<Kokkos::pair<size_t, size_t>*>::const_type row_range;
    Kokkos::View<FreedomSignature*>::const_type base_node_freedom_signature;
    Kokkos::View<FreedomSignature*>::const_type target_node_freedom_signature;

    KOKKOS_FUNCTION void operator()(int i_constraint, size_t& update) const {
        const auto num_rows = row_range(i_constraint).second - row_range(i_constraint).first;
        const auto num_base_cols = count_active_dofs(base_node_freedom_signature(i_constraint));
        const auto num_target_cols = count_active_dofs(target_node_freedom_signature(i_constraint));
        update += num_rows * (num_base_cols + num_target_cols);
    }
};

}  // namespace openturbine
