#pragma once

#include <Kokkos_Core.hpp>

#include "constraints/constraint_type.hpp"

namespace openturbine {

/// ComputeNumberOfNonZeros_Constraints calculates the total number of nonzero values in the
/// constraint gradient matrix based on the block layout of the constraints.
struct ComputeNumberOfNonZeros_Constraints {
    Kokkos::View<Kokkos::pair<size_t, size_t>*>::const_type row_range;
    Kokkos::View<Kokkos::pair<size_t, size_t>*>::const_type base_node_col_range;
    Kokkos::View<Kokkos::pair<size_t, size_t>*>::const_type target_node_col_range;

    KOKKOS_FUNCTION void operator()(int i_constraint, size_t& update) const {
        const auto num_rows = row_range(i_constraint).second - row_range(i_constraint).first;
        const auto num_base_cols =
            base_node_col_range(i_constraint).second - base_node_col_range(i_constraint).first;
        const auto num_target_cols =
            target_node_col_range(i_constraint).second - target_node_col_range(i_constraint).first;
        update += num_rows * (num_base_cols + num_target_cols);
    }
};

}  // namespace openturbine
