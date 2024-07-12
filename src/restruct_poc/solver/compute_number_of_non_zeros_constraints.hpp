#pragma once

#include <Kokkos_Core.hpp>

#include "src/restruct_poc/solver/constraints.hpp"

namespace openturbine {
struct ComputeNumberOfNonZeros_Constraints {
    Kokkos::View<Constraints::DeviceData*>::const_type constraint_data;

    KOKKOS_FUNCTION
    void operator()(int i_constraint, int& update) const {
        auto& cd = constraint_data[i_constraint];
        auto num_blocks = cd.base_node_index < 0 ? 1 : 2;
        update += num_blocks * kLieAlgebraComponents * (cd.row_range.second - cd.row_range.first);
    }
};
}  // namespace openturbine