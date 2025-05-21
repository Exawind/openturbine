#pragma once

#include <Kokkos_Core.hpp>

#include "freedom_signature.hpp"
#include "state/state.hpp"

namespace openturbine {

template <typename DeviceType>
struct ComputeNodeFreedomMapTable {
    typename Kokkos::View<FreedomSignature*, DeviceType>::const_type node_freedom_allocation_table;
    Kokkos::View<size_t*, DeviceType> node_freedom_map_table;

    KOKKOS_FUNCTION
    void operator()(size_t i, size_t& update, bool is_final) const {
        const auto num_dof = count_active_dofs(node_freedom_allocation_table(i));
        update += num_dof;
        if (is_final) {
            node_freedom_map_table(i + 1) = update;
        }
    }
};

template <typename DeviceType>
inline void compute_node_freedom_map_table(State<DeviceType>& state) {
    Kokkos::deep_copy(state.node_freedom_map_table, 0UL);
    auto result = 0UL;
    auto scan_range =
        Kokkos::RangePolicy<typename DeviceType::execution_space>(0, state.num_system_nodes - 1UL);
    Kokkos::parallel_scan(
        "Compute Node Freedom Map Table", scan_range,
        ComputeNodeFreedomMapTable<DeviceType>{
            state.node_freedom_allocation_table, state.node_freedom_map_table
        },
        result
    );
}

}  // namespace openturbine
