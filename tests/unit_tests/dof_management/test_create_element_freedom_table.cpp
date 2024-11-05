#include <gtest/gtest.h>

#include "src/dof_management/create_element_freedom_table.hpp"

namespace openturbine::tests {

TEST(TestCreateElementFreedomTable, OneBeamOneNode) {
    auto state = State(1U);
    Kokkos::deep_copy(state.node_freedom_map_table, 0U);

    auto beams = Beams(1U, 1U, 1U);
    Kokkos::deep_copy(beams.node_state_indices, 0U);
    Kokkos::deep_copy(beams.num_nodes_per_element, 1U);

    create_element_freedom_table(beams, state);

    const auto host_element_freedom_table = Kokkos::create_mirror(beams.element_freedom_table);
    Kokkos::deep_copy(host_element_freedom_table, beams.element_freedom_table);

    for (auto k = 0U; k < 6U; ++k) {
        EXPECT_EQ(host_element_freedom_table(0, 0, k), k);
    }
}

TEST(TestCreateElementFreedomTable, OneBeamTwoNodes) {
    auto state = State(2U);
    constexpr auto host_node_freedom_map_table_data = std::array{0UL, 6UL};
    const auto host_node_freedom_map_table = Kokkos::View<size_t[2], Kokkos::HostSpace>::const_type(
        host_node_freedom_map_table_data.data()
    );
    const auto mirror_node_freedom_map_table = Kokkos::create_mirror(state.node_freedom_map_table);
    Kokkos::deep_copy(mirror_node_freedom_map_table, host_node_freedom_map_table);
    Kokkos::deep_copy(state.node_freedom_map_table, mirror_node_freedom_map_table);

    auto beams = Beams(1U, 2U, 1U);
    constexpr auto host_node_state_indices_data = std::array{0UL, 1UL};
    const auto host_node_state_indices =
        Kokkos::View<size_t[1][2], Kokkos::HostSpace>::const_type(host_node_state_indices_data.data()
        );
    const auto mirror_node_state_indices = Kokkos::create_mirror(beams.node_state_indices);
    Kokkos::deep_copy(mirror_node_state_indices, host_node_state_indices);
    Kokkos::deep_copy(beams.node_state_indices, mirror_node_state_indices);
    Kokkos::deep_copy(beams.num_nodes_per_element, 2U);

    create_element_freedom_table(beams, state);

    const auto host_element_freedom_table = Kokkos::create_mirror(beams.element_freedom_table);
    Kokkos::deep_copy(host_element_freedom_table, beams.element_freedom_table);

    for (auto k = 0U; k < 6U; ++k) {
        EXPECT_EQ(host_element_freedom_table(0, 0, k), k);
    }
    for (auto k = 0U; k < 6U; ++k) {
        EXPECT_EQ(host_element_freedom_table(0, 1, k), k + 6U);
    }
}

TEST(TestCreateElementFreedomTable, TwoBeamsOneNode) {
    auto state = State(2U);
    constexpr auto host_node_freedom_map_table_data = std::array{0UL, 6UL};
    const auto host_node_freedom_map_table = Kokkos::View<size_t[2], Kokkos::HostSpace>::const_type(
        host_node_freedom_map_table_data.data()
    );
    const auto mirror_node_freedom_map_table = Kokkos::create_mirror(state.node_freedom_map_table);
    Kokkos::deep_copy(mirror_node_freedom_map_table, host_node_freedom_map_table);
    Kokkos::deep_copy(state.node_freedom_map_table, mirror_node_freedom_map_table);

    auto beams = Beams(2U, 1U, 1U);
    constexpr auto host_node_state_indices_data = std::array{0UL, 1UL};
    const auto host_node_state_indices =
        Kokkos::View<size_t[2][1], Kokkos::HostSpace>::const_type(host_node_state_indices_data.data()
        );
    const auto mirror_node_state_indices = Kokkos::create_mirror(beams.node_state_indices);
    Kokkos::deep_copy(mirror_node_state_indices, host_node_state_indices);
    Kokkos::deep_copy(beams.node_state_indices, mirror_node_state_indices);
    Kokkos::deep_copy(beams.num_nodes_per_element, 1U);

    create_element_freedom_table(beams, state);

    const auto host_element_freedom_table = Kokkos::create_mirror(beams.element_freedom_table);
    Kokkos::deep_copy(host_element_freedom_table, beams.element_freedom_table);

    for (auto k = 0U; k < 6U; ++k) {
        EXPECT_EQ(host_element_freedom_table(0, 0, k), k);
    }
    for (auto k = 0U; k < 6U; ++k) {
        EXPECT_EQ(host_element_freedom_table(1, 0, k), k + 6U);
    }
}

}  // namespace openturbine::tests
