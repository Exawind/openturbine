#include <gtest/gtest.h>

#include "src/dof_management/create_element_freedom_table.hpp"

namespace openturbine::tests {

TEST(TestCreateElementFreedomTable, OneBeamElementWithOneNode_NoMassElement) {
    auto state = State(1U);
    Kokkos::deep_copy(state.node_freedom_map_table, 0U);

    auto beams = std::make_shared<Beams>(1U, 1U, 1U);
    Kokkos::deep_copy(beams->node_state_indices, 0U);
    Kokkos::deep_copy(beams->num_nodes_per_element, 1U);
    auto elements = Elements{beams};  // 1 beam element + 0 mass elements

    EXPECT_EQ(elements.NumElementsInSystem(), 1);

    create_element_freedom_table(elements, state);

    const auto host_element_freedom_table = Kokkos::create_mirror(beams->element_freedom_table);
    Kokkos::deep_copy(host_element_freedom_table, beams->element_freedom_table);

    for (auto k = 0U; k < 6U; ++k) {
        EXPECT_EQ(
            host_element_freedom_table(0, 0, k), k
        );  // Beam Element 1 Node 1 DOFs: 0, 1, 2, 3, 4, 5
    }
}

TEST(TestCreateElementFreedomTable, NoBeamElement_OneMassElementWithOneNode) {
    auto state = State(1U);
    Kokkos::deep_copy(state.node_freedom_map_table, 0U);

    auto masses = std::make_shared<Masses>(1U);
    Kokkos::deep_copy(masses->state_indices, 0U);
    auto elements = Elements{nullptr, masses};  // 0 beam elements + 1 mass element

    EXPECT_EQ(elements.NumElementsInSystem(), 1);

    create_element_freedom_table(elements, state);

    const auto host_element_freedom_table = Kokkos::create_mirror(masses->element_freedom_table);
    Kokkos::deep_copy(host_element_freedom_table, masses->element_freedom_table);

    for (auto k = 0U; k < 6U; ++k) {
        EXPECT_EQ(
            host_element_freedom_table(0, 0, k), k
        );  // Mass Element 1 Node 1 DOFs: 0, 1, 2, 3, 4, 5
    }
}

TEST(TestCreateElementFreedomTable, OneBeamElementWithTwoNodes_NoMassElement) {
    auto state = State(2U);
    constexpr auto host_node_freedom_map_table_data = std::array{0UL, 6UL};
    const auto host_node_freedom_map_table = Kokkos::View<size_t[2], Kokkos::HostSpace>::const_type(
        host_node_freedom_map_table_data.data()
    );
    const auto mirror_node_freedom_map_table = Kokkos::create_mirror(state.node_freedom_map_table);
    Kokkos::deep_copy(mirror_node_freedom_map_table, host_node_freedom_map_table);
    Kokkos::deep_copy(state.node_freedom_map_table, mirror_node_freedom_map_table);

    auto beams = std::make_shared<Beams>(1U, 2U, 1U);
    constexpr auto host_node_state_indices_data = std::array{0UL, 1UL};
    const auto host_node_state_indices =
        Kokkos::View<size_t[1][2], Kokkos::HostSpace>::const_type(host_node_state_indices_data.data()
        );
    const auto mirror_node_state_indices = Kokkos::create_mirror(beams->node_state_indices);
    Kokkos::deep_copy(mirror_node_state_indices, host_node_state_indices);
    Kokkos::deep_copy(beams->node_state_indices, mirror_node_state_indices);
    Kokkos::deep_copy(beams->num_nodes_per_element, 2U);
    auto elements = Elements{beams};  // 1 beam element + 0 mass elements

    EXPECT_EQ(elements.NumElementsInSystem(), 1);

    create_element_freedom_table(elements, state);

    const auto host_element_freedom_table = Kokkos::create_mirror(beams->element_freedom_table);
    Kokkos::deep_copy(host_element_freedom_table, beams->element_freedom_table);

    for (auto k = 0U; k < 6U; ++k) {
        EXPECT_EQ(
            host_element_freedom_table(0, 0, k), k
        );  // Beam Element 1 Node 1 DOFs: 0, 1, 2, 3, 4, 5
    }
    for (auto k = 0U; k < 6U; ++k) {
        EXPECT_EQ(
            host_element_freedom_table(0, 1, k), k + 6U
        );  // Beam Element 1 Node 2 DOFs: 6, 7, 8, 9, 10, 11
    }
}

TEST(TestCreateElementFreedomTable, OneBeamElementWithOneNode_OneMassElementWithOneNode) {
    auto state = State(2U);
    constexpr auto host_node_freedom_map_table_data = std::array{0UL, 6UL};
    const auto host_node_freedom_map_table = Kokkos::View<size_t[2], Kokkos::HostSpace>::const_type(
        host_node_freedom_map_table_data.data()
    );
    const auto mirror_node_freedom_map_table = Kokkos::create_mirror(state.node_freedom_map_table);
    Kokkos::deep_copy(mirror_node_freedom_map_table, host_node_freedom_map_table);
    Kokkos::deep_copy(state.node_freedom_map_table, mirror_node_freedom_map_table);

    auto beams = std::make_shared<Beams>(1U, 1U, 1U);
    Kokkos::deep_copy(beams->node_state_indices, 0U);
    Kokkos::deep_copy(beams->num_nodes_per_element, 1U);
    auto masses = std::make_shared<Masses>(1U);
    Kokkos::deep_copy(masses->state_indices, 1U);
    auto elements = Elements{beams, masses};  // 1 beam element + 1 mass element

    EXPECT_EQ(elements.NumElementsInSystem(), 2);

    create_element_freedom_table(elements, state);

    const auto host_beams_element_freedom_table =
        Kokkos::create_mirror(beams->element_freedom_table);
    Kokkos::deep_copy(host_beams_element_freedom_table, beams->element_freedom_table);
    for (auto k = 0U; k < 6U; ++k) {
        EXPECT_EQ(
            host_beams_element_freedom_table(0, 0, k), k
        );  // Beam Element 1 Node 1 DOFs: 0, 1, 2, 3, 4, 5
    }

    const auto host_masses_element_freedom_table =
        Kokkos::create_mirror(masses->element_freedom_table);
    Kokkos::deep_copy(host_masses_element_freedom_table, masses->element_freedom_table);
    for (auto k = 0U; k < 6U; ++k) {
        EXPECT_EQ(
            host_masses_element_freedom_table(0, 0, k), k + 6U
        );  // Mass Element 1 Node 1 DOFs: 6, 7, 8, 9, 10, 11
    }
}

TEST(TestCreateElementFreedomTable, TwoBeamElementsWithOneNode_NoMassElement) {
    auto state = State(2U);
    constexpr auto host_node_freedom_map_table_data = std::array{0UL, 6UL};
    const auto host_node_freedom_map_table = Kokkos::View<size_t[2], Kokkos::HostSpace>::const_type(
        host_node_freedom_map_table_data.data()
    );
    const auto mirror_node_freedom_map_table = Kokkos::create_mirror(state.node_freedom_map_table);
    Kokkos::deep_copy(mirror_node_freedom_map_table, host_node_freedom_map_table);
    Kokkos::deep_copy(state.node_freedom_map_table, mirror_node_freedom_map_table);

    auto beams = std::make_shared<Beams>(2U, 1U, 1U);
    constexpr auto host_node_state_indices_data = std::array{0UL, 1UL};
    const auto host_node_state_indices =
        Kokkos::View<size_t[2][1], Kokkos::HostSpace>::const_type(host_node_state_indices_data.data()
        );
    const auto mirror_node_state_indices = Kokkos::create_mirror(beams->node_state_indices);
    Kokkos::deep_copy(mirror_node_state_indices, host_node_state_indices);
    Kokkos::deep_copy(beams->node_state_indices, mirror_node_state_indices);
    Kokkos::deep_copy(beams->num_nodes_per_element, 1U);
    auto elements = Elements{beams};  // 2 beam elements + 0 mass elements

    EXPECT_EQ(elements.NumElementsInSystem(), 2);

    create_element_freedom_table(elements, state);

    const auto host_element_freedom_table = Kokkos::create_mirror(beams->element_freedom_table);
    Kokkos::deep_copy(host_element_freedom_table, beams->element_freedom_table);

    for (auto k = 0U; k < 6U; ++k) {
        EXPECT_EQ(
            host_element_freedom_table(0, 0, k), k
        );  // Beam Element 1 Node 1 DOFs: 0, 1, 2, 3, 4, 5
    }
    for (auto k = 0U; k < 6U; ++k) {
        EXPECT_EQ(
            host_element_freedom_table(1, 0, k), k + 6U
        );  // Beam Element 2 Node 1 DOFs: 6, 7, 8, 9, 10, 11
    }
}

TEST(TestCreateElementFreedomTable, TwoBeamElementsWithTwoNodesShared_NoMassElement) {
    auto state = State(3U);
    constexpr auto host_node_freedom_map_table_data = std::array{0UL, 6UL, 12UL};
    const auto host_node_freedom_map_table = Kokkos::View<size_t[3], Kokkos::HostSpace>::const_type(
        host_node_freedom_map_table_data.data()
    );
    const auto mirror_node_freedom_map_table = Kokkos::create_mirror(state.node_freedom_map_table);
    Kokkos::deep_copy(mirror_node_freedom_map_table, host_node_freedom_map_table);
    Kokkos::deep_copy(state.node_freedom_map_table, mirror_node_freedom_map_table);

    auto beams = std::make_shared<Beams>(2U, 2U, 1U);
    constexpr auto host_node_state_indices_data = std::array{0UL, 1UL, 1UL, 2UL};
    const auto host_node_state_indices =
        Kokkos::View<size_t[2][2], Kokkos::HostSpace>::const_type(host_node_state_indices_data.data()
        );
    const auto mirror_node_state_indices = Kokkos::create_mirror(beams->node_state_indices);
    Kokkos::deep_copy(mirror_node_state_indices, host_node_state_indices);
    Kokkos::deep_copy(beams->node_state_indices, mirror_node_state_indices);
    Kokkos::deep_copy(beams->num_nodes_per_element, 2U);
    auto elements = Elements{beams};  // 2 beam elements + 0 mass elements

    EXPECT_EQ(elements.NumElementsInSystem(), 2);

    create_element_freedom_table(elements, state);

    const auto host_element_freedom_table = Kokkos::create_mirror(beams->element_freedom_table);
    Kokkos::deep_copy(host_element_freedom_table, beams->element_freedom_table);

    for (auto k = 0U; k < 6U; ++k) {
        EXPECT_EQ(
            host_element_freedom_table(0, 0, k), k
        );  // Beam Element 1 Node 1 DOFs: 0, 1, 2, 3, 4, 5
    }
    for (auto k = 0U; k < 6U; ++k) {
        EXPECT_EQ(
            host_element_freedom_table(0, 1, k), k + 6U
        );  // Beam Element 1 Node 2 DOFs: 6, 7, 8, 9, 10, 11
    }
    for (auto k = 0U; k < 6U; ++k) {
        EXPECT_EQ(
            host_element_freedom_table(1, 0, k), k + 6U
        );  // Beam Element 2 Node 1 DOFs: 6, 7, 8, 9, 10, 11
    }
    for (auto k = 0U; k < 6U; ++k) {
        EXPECT_EQ(
            host_element_freedom_table(1, 1, k), k + 12U
        );  // Beam Element 2 Node 2 DOFs: 12, 13, 14, 15, 16, 17
    }
}

TEST(TestCreateElementFreedomTable, TwoBeamElementsWithTwoNodesShared_Flipped_NoMassElement) {
    auto state = State(3U);
    constexpr auto host_node_freedom_map_table_data = std::array{0UL, 6UL, 12UL};
    const auto host_node_freedom_map_table = Kokkos::View<size_t[3], Kokkos::HostSpace>::const_type(
        host_node_freedom_map_table_data.data()
    );
    const auto mirror_node_freedom_map_table = Kokkos::create_mirror(state.node_freedom_map_table);
    Kokkos::deep_copy(mirror_node_freedom_map_table, host_node_freedom_map_table);
    Kokkos::deep_copy(state.node_freedom_map_table, mirror_node_freedom_map_table);

    auto beams = std::make_shared<Beams>(2U, 2U, 1U);
    constexpr auto host_node_state_indices_data = std::array{1UL, 2UL, 0UL, 1UL};
    const auto host_node_state_indices =
        Kokkos::View<size_t[2][2], Kokkos::HostSpace>::const_type(host_node_state_indices_data.data()
        );
    const auto mirror_node_state_indices = Kokkos::create_mirror(beams->node_state_indices);
    Kokkos::deep_copy(mirror_node_state_indices, host_node_state_indices);
    Kokkos::deep_copy(beams->node_state_indices, mirror_node_state_indices);
    Kokkos::deep_copy(beams->num_nodes_per_element, 2U);
    auto elements = Elements{beams};  // 2 beam elements + 0 mass elements

    EXPECT_EQ(elements.NumElementsInSystem(), 2);

    create_element_freedom_table(elements, state);

    const auto host_element_freedom_table = Kokkos::create_mirror(beams->element_freedom_table);
    Kokkos::deep_copy(host_element_freedom_table, beams->element_freedom_table);

    for (auto k = 0U; k < 6U; ++k) {
        EXPECT_EQ(
            host_element_freedom_table(1, 0, k), k
        );  // Beam Element 2 Node 1 DOFs: 0, 1, 2, 3, 4, 5
    }
    for (auto k = 0U; k < 6U; ++k) {
        EXPECT_EQ(
            host_element_freedom_table(1, 1, k), k + 6U
        );  // Beam Element 2 Node 2 DOFs: 6, 7, 8, 9, 10, 11
    }
    for (auto k = 0U; k < 6U; ++k) {
        EXPECT_EQ(
            host_element_freedom_table(0, 0, k), k + 6U
        );  // Beam Element 1 Node 1 DOFs: 6, 7, 8, 9, 10, 11
    }
    for (auto k = 0U; k < 6U; ++k) {
        EXPECT_EQ(
            host_element_freedom_table(0, 1, k), k + 12U
        );  // Beam Element 1 Node 2 DOFs: 12, 13, 14, 15, 16, 17
    }
}

TEST(TestCreateElementFreedomTable, TwoBeamElementsWithOneNode_OneMassElementWithOneNode) {
    auto state = State(3U);
    constexpr auto host_node_freedom_map_table_data = std::array{0UL, 6UL, 12UL};
    const auto host_node_freedom_map_table = Kokkos::View<size_t[3], Kokkos::HostSpace>::const_type(
        host_node_freedom_map_table_data.data()
    );
    const auto mirror_node_freedom_map_table = Kokkos::create_mirror(state.node_freedom_map_table);
    Kokkos::deep_copy(mirror_node_freedom_map_table, host_node_freedom_map_table);
    Kokkos::deep_copy(state.node_freedom_map_table, mirror_node_freedom_map_table);

    auto beams = std::make_shared<Beams>(2U, 1U, 1U);
    constexpr auto host_node_state_indices_data = std::array{0UL, 1UL};
    const auto host_node_state_indices =
        Kokkos::View<size_t[2][1], Kokkos::HostSpace>::const_type(host_node_state_indices_data.data()
        );
    const auto mirror_node_state_indices = Kokkos::create_mirror(beams->node_state_indices);
    Kokkos::deep_copy(mirror_node_state_indices, host_node_state_indices);
    Kokkos::deep_copy(beams->node_state_indices, mirror_node_state_indices);
    Kokkos::deep_copy(beams->num_nodes_per_element, 1U);
    auto masses = std::make_shared<Masses>(1U);
    Kokkos::deep_copy(masses->state_indices, 2U);
    auto elements = Elements{beams, masses};  // 2 beam elements + 1 mass element

    EXPECT_EQ(elements.NumElementsInSystem(), 3);

    create_element_freedom_table(elements, state);

    const auto host_beams_element_freedom_table =
        Kokkos::create_mirror(beams->element_freedom_table);
    Kokkos::deep_copy(host_beams_element_freedom_table, beams->element_freedom_table);
    for (auto k = 0U; k < 6U; ++k) {
        EXPECT_EQ(
            host_beams_element_freedom_table(0, 0, k), k
        );  // Beam Element 1 Node 1 DOFs: 0, 1, 2, 3, 4, 5
    }
    for (auto k = 0U; k < 6U; ++k) {
        EXPECT_EQ(
            host_beams_element_freedom_table(1, 0, k), k + 6U
        );  // Beam Element 2 Node 1 DOFs: 6, 7, 8, 9, 10, 11
    }

    const auto host_masses_element_freedom_table =
        Kokkos::create_mirror(masses->element_freedom_table);
    Kokkos::deep_copy(host_masses_element_freedom_table, masses->element_freedom_table);
    for (auto k = 0U; k < 6U; ++k) {
        EXPECT_EQ(
            host_masses_element_freedom_table(0, 0, k), k + 12U
        );  // Mass Element 1 Node 1 DOFs: 12, 13, 14, 15, 16, 17
    }
}

}  // namespace openturbine::tests
