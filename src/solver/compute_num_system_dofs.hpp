#pragma once

#include <Kokkos_Core.hpp>

namespace openturbine {

template <typename DeviceType>
struct ComputeNumSystemDofsReducer {
    typename Kokkos::View<size_t*, DeviceType>::const_type active_dofs;

    KOKKOS_FUNCTION
    void operator()(size_t i, size_t& update) const { update += active_dofs(i); }
};

/// Computes the total number of active degrees of freedom in the system
template <typename DeviceType>
[[nodiscard]] inline size_t ComputeNumSystemDofs(
    const typename Kokkos::View<size_t*, DeviceType>::const_type& active_dofs
) {
    auto total_system_dofs = 0UL;

    using RangePolicy = Kokkos::RangePolicy<typename DeviceType::execution_space>;

    Kokkos::parallel_reduce(
        "ComputeNumSystemDofs", RangePolicy(0, active_dofs.extent(0)),
        ComputeNumSystemDofsReducer<DeviceType>{active_dofs}, total_system_dofs
    );
    return total_system_dofs;
}

}  // namespace openturbine
