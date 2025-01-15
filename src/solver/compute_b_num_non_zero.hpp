#pragma once

#include <Kokkos_Core.hpp>

#include "compute_number_of_non_zeros_constraints.hpp"

namespace openturbine {
[[nodiscard]] static size_t ComputeBNumNonZero(
    const Kokkos::View<ConstraintType*>::const_type& constraint_type,
    const Kokkos::View<Kokkos::pair<size_t, size_t>*>::const_type& constraint_row_range
) {
    auto B_num_non_zero = size_t{0U};
    Kokkos::parallel_reduce(
        "ComputeNumberOfNonZeros_Constraints", constraint_type.extent(0),
        ComputeNumberOfNonZeros_Constraints{constraint_type, constraint_row_range}, B_num_non_zero
    );
    return B_num_non_zero;
}
}  // namespace openturbine
