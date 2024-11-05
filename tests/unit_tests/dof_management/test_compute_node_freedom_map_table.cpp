#include <gtest/gtest.h>

#include "src/dof_management/compute_node_freedom_map_table.hpp"

namespace openturbine::tests {

TEST(TestAssembleNodeFreedomMapTable, OneNode) {
    auto state = State(1U);
    Kokkos::deep_copy(state.node_freedom_allocation_table, FreedomSignature::AllComponents);

    compute_node_freedom_map_table(state);

    const auto host_node_freedom_map_table = Kokkos::create_mirror(state.node_freedom_map_table);
    Kokkos::deep_copy(host_node_freedom_map_table, state.node_freedom_map_table);

    EXPECT_EQ(host_node_freedom_map_table(0), 0);
}

TEST(TestAssembleNodeFreedomMapTable, FourNodes) {
    auto state = State(4U);
    constexpr auto host_node_freedom_allocation_table_data = std::array{
        FreedomSignature::AllComponents, FreedomSignature::JustPosition,
        FreedomSignature::JustRotation, FreedomSignature::NoComponents
    };
    const auto host_node_freedom_allocation_table =
        Kokkos::View<FreedomSignature[4], Kokkos::HostSpace>::const_type(
            host_node_freedom_allocation_table_data.data()
        );
    const auto mirror_node_freedom_allocation_table =
        Kokkos::create_mirror(state.node_freedom_allocation_table);
    Kokkos::deep_copy(mirror_node_freedom_allocation_table, host_node_freedom_allocation_table);
    Kokkos::deep_copy(state.node_freedom_allocation_table, mirror_node_freedom_allocation_table);

    compute_node_freedom_map_table(state);

    const auto host_node_freedom_map_table = Kokkos::create_mirror(state.node_freedom_map_table);
    Kokkos::deep_copy(host_node_freedom_map_table, state.node_freedom_map_table);

    EXPECT_EQ(host_node_freedom_map_table(0), 0);
    EXPECT_EQ(host_node_freedom_map_table(1), 7);
    EXPECT_EQ(host_node_freedom_map_table(2), 10);
    EXPECT_EQ(host_node_freedom_map_table(3), 14);
}

}  // namespace openturbine::tests
