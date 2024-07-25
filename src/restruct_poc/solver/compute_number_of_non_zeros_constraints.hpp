#pragma once

#include <Kokkos_Core.hpp>

#include "src/restruct_poc/solver/constraints.hpp"

namespace openturbine {

/// ComputeNumberOfNonZeros_Constraints calculates the total number of nonzero values in the
/// constraint gradient matrix based on the block layout of the constraints.
struct ComputeNumberOfNonZeros_Constraints {
    Kokkos::View<Constraints::DeviceData*>::const_type constraint_data;

    KOKKOS_FUNCTION
    void operator()(int i_constraint, size_t& update) const {
        auto& cd = constraint_data[i_constraint];
        auto num_blocks = cd.base_node_index < 0 ? 1u : 2u;
        update += num_blocks * kLieAlgebraComponents * (cd.row_range.second - cd.row_range.first);
    }
};
}  // namespace openturbine