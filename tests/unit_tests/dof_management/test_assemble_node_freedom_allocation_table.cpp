#include <gtest/gtest.h>

#include "src/dof_management/assemble_node_freedom_allocation_table.hpp"

namespace openturbine::tests {

TEST(TestAssembleNodeFreedomAllocationTable, OneBeamElementWithOneNode_NoMassElement) {
    auto state = State(1U);  // 1 node in the system

    auto beams = std::make_shared<Beams>(1U, 1U, 1U);  // 1 beam element with 1 node per element
    Kokkos::deep_copy(beams->node_state_indices, 0U);
    Kokkos::deep_copy(beams->num_nodes_per_element, 1U);
    auto elements = Elements{beams};  // No mass elements in the model

    auto constraints = Constraints(std::vector<std::shared_ptr<Constraint>>{});

    assemble_node_freedom_allocation_table(state, elements, constraints);

    const auto host_node_freedom_allocation_table =
        Kokkos::create_mirror(state.node_freedom_allocation_table);
    Kokkos::deep_copy(host_node_freedom_allocation_table, state.node_freedom_allocation_table);

    EXPECT_EQ(host_node_freedom_allocation_table(0), FreedomSignature::AllComponents);
}

TEST(TestAssembleNodeFreedomAllocationTable, OneBeamElementWithOneNode_OneMassElementWithOneNode) {
    auto state = State(2U);  // 2 nodes in the system

    auto beams = std::make_shared<Beams>(1U, 1U, 1U);  // 1 beam element with 1 node per element
    Kokkos::deep_copy(beams->node_state_indices, 0U);
    Kokkos::deep_copy(beams->num_nodes_per_element, 1U);
    auto masses = std::make_shared<Masses>(1U);  // 1 mass element with 1 node
    Kokkos::deep_copy(masses->state_indices, 1U);
    auto elements = Elements{beams, masses};

    auto constraints = Constraints(std::vector<std::shared_ptr<Constraint>>{});

    assemble_node_freedom_allocation_table(state, elements, constraints);

    const auto host_node_freedom_allocation_table =
        Kokkos::create_mirror(state.node_freedom_allocation_table);
    Kokkos::deep_copy(host_node_freedom_allocation_table, state.node_freedom_allocation_table);

    EXPECT_EQ(host_node_freedom_allocation_table(0), FreedomSignature::AllComponents);
    EXPECT_EQ(host_node_freedom_allocation_table(1), FreedomSignature::AllComponents);
}

TEST(TestAssembleNodeFreedomAllocationTable, NoBeamElements_OneMassElementWithOneNode) {
    auto state = State(1U);  // 1 node in the system

    auto masses = std::make_shared<Masses>(1U);  // 1 mass element with 1 node
    Kokkos::deep_copy(masses->state_indices, 0U);
    auto elements = Elements{nullptr, masses};  // No beam elements in the model

    auto constraints = Constraints(std::vector<std::shared_ptr<Constraint>>{});

    assemble_node_freedom_allocation_table(state, elements, constraints);

    const auto host_node_freedom_allocation_table =
        Kokkos::create_mirror(state.node_freedom_allocation_table);
    Kokkos::deep_copy(host_node_freedom_allocation_table, state.node_freedom_allocation_table);

    EXPECT_EQ(host_node_freedom_allocation_table(0), FreedomSignature::AllComponents);
}

TEST(TestAssembleNodeFreedomAllocationTable, OneBeamElementWithTwoNodes_NoMassElement) {
    auto state = State(2U);  // 2 nodes in the system

    auto beams = std::make_shared<Beams>(1U, 2U, 1U);  // 1 beam element with 2 nodes per element
    constexpr auto host_node_state_indices_data = std::array{0UL, 1UL};
    const auto host_node_state_indices =
        Kokkos::View<size_t[1][2], Kokkos::HostSpace>::const_type(host_node_state_indices_data.data()
        );
    const auto mirror_node_state_indices = Kokkos::create_mirror(beams->node_state_indices);
    Kokkos::deep_copy(mirror_node_state_indices, host_node_state_indices);
    Kokkos::deep_copy(beams->node_state_indices, mirror_node_state_indices);
    Kokkos::deep_copy(beams->num_nodes_per_element, 2U);
    auto elements = Elements{beams};  // No mass elements in the model

    auto constraints = Constraints(std::vector<std::shared_ptr<Constraint>>{});

    assemble_node_freedom_allocation_table(state, elements, constraints);

    const auto host_node_freedom_allocation_table =
        Kokkos::create_mirror(state.node_freedom_allocation_table);
    Kokkos::deep_copy(host_node_freedom_allocation_table, state.node_freedom_allocation_table);

    EXPECT_EQ(host_node_freedom_allocation_table(0), FreedomSignature::AllComponents);
    EXPECT_EQ(host_node_freedom_allocation_table(1), FreedomSignature::AllComponents);
}

TEST(TestAssembleNodeFreedomAllocationTable, TwoBeamElementsWithOneNode_NoMassElement) {
    auto state = State(2U);  // 2 nodes in the system

    auto beams = std::make_shared<Beams>(2U, 1U, 1U);  // 2 beam elements with 1 node per element
    constexpr auto host_node_state_indices_data = std::array{0UL, 1UL};
    const auto host_node_state_indices =
        Kokkos::View<size_t[2][1], Kokkos::HostSpace>::const_type(host_node_state_indices_data.data()
        );
    const auto mirror_node_state_indices = Kokkos::create_mirror(beams->node_state_indices);
    Kokkos::deep_copy(mirror_node_state_indices, host_node_state_indices);
    Kokkos::deep_copy(beams->node_state_indices, mirror_node_state_indices);
    Kokkos::deep_copy(beams->num_nodes_per_element, 1U);
    auto elements = Elements{beams};  // No mass elements in the model

    auto constraints = Constraints(std::vector<std::shared_ptr<Constraint>>{});

    assemble_node_freedom_allocation_table(state, elements, constraints);

    const auto host_node_freedom_allocation_table =
        Kokkos::create_mirror(state.node_freedom_allocation_table);
    Kokkos::deep_copy(host_node_freedom_allocation_table, state.node_freedom_allocation_table);

    EXPECT_EQ(host_node_freedom_allocation_table(0), FreedomSignature::AllComponents);
    EXPECT_EQ(host_node_freedom_allocation_table(1), FreedomSignature::AllComponents);
}

}  // namespace openturbine::tests
