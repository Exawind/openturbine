#pragma once

#include <Kokkos_Core.hpp>

#include "src/constraints/constraint_type.hpp"

namespace openturbine {

/// ComputeNumberOfNonZeros_Constraints calculates the total number of nonzero values in the
/// constraint gradient matrix based on the block layout of the constraints.
struct ComputeNumberOfNonZeros_Constraints {
    Kokkos::View<ConstraintType*>::const_type type;
    Kokkos::View<Kokkos::pair<size_t, size_t>*>::const_type row_range;

    KOKKOS_FUNCTION
    void operator()(int i_constraint, size_t& update) const {
        const auto num_blocks = GetNumberOfNodes(type(i_constraint));
        update += num_blocks * 6U * (row_range(i_constraint).second - row_range(i_constraint).first);
    }
};
}  // namespace openturbine
