#pragma once

#include <Kokkos_Core.hpp>

#include "compute_number_of_non_zeros_constraints.hpp"
#include "constraints/constraint_type.hpp"

namespace openturbine {
[[nodiscard]] static size_t ComputeBNumNonZero(
    const Kokkos::View<Kokkos::pair<size_t, size_t>*>::const_type& row_range,
    const Kokkos::View<FreedomSignature*>::const_type& base_node_freedom_signature,
    const Kokkos::View<FreedomSignature*>::const_type& target_node_freedom_signature
) {
    auto B_num_non_zero = size_t{0U};
    Kokkos::parallel_reduce(
        "ComputeNumberOfNonZeros_Constraints", row_range.extent(0),
        ComputeNumberOfNonZeros_Constraints{
            row_range,
            base_node_freedom_signature,
            target_node_freedom_signature,
        },
        B_num_non_zero
    );
    return B_num_non_zero;
}
}  // namespace openturbine
