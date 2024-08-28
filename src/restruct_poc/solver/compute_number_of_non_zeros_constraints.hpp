#pragma once

#include <Kokkos_Core.hpp>

#include "src/restruct_poc/solver/constraints.hpp"

namespace openturbine {

/// ComputeNumberOfNonZeros_Constraints calculates the total number of nonzero values in the
/// constraint gradient matrix based on the block layout of the constraints.
struct ComputeNumberOfNonZeros_Constraints {
    Kokkos::View<ConstraintType*>::const_type type;
    Kokkos::View<size_t*[2]>::const_type row_range;

    KOKKOS_FUNCTION
    void operator()(int i_constraint, size_t& update) const {
        const auto num_blocks = GetNumberOfNodes(type(i_constraint));
        update += num_blocks * kLieAlgebraComponents * (row_range(i_constraint, 1) - row_range(i_constraint, 0));
    }
};
}  // namespace openturbine
