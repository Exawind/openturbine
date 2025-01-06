#include <gtest/gtest.h>

#include "src/dof_management/assemble_node_freedom_allocation_table.hpp"
#include "src/dof_management/compute_node_freedom_map_table.hpp"
#include "src/dof_management/create_constraint_freedom_table.hpp"

namespace openturbine::tests {

TEST(TestCreateConstraintFreedomTable, SingleNodeConstraint_FixedBC) {
    auto invalid_node = Node(0U, {0., 0., 0., 1., 0., 0., 0.});  // base node - index is 0
    auto node_1 = Node(1U, {1., 0., 0., 1., 0., 0., 0.});        // target node - index is 1
    auto fixed_bc = std::make_shared<Constraint>(ConstraintType::kFixedBC, 1, invalid_node, node_1);
    auto constraints = Constraints(std::vector<std::shared_ptr<Constraint>>{fixed_bc});

    auto state = State(2U);  // 2 nodes in the system
    constexpr auto host_node_freedom_map_table_data = std::array{0UL, 6UL};
    const auto host_node_freedom_map_table = Kokkos::View<size_t[2], Kokkos::HostSpace>::const_type(
        host_node_freedom_map_table_data.data()
    );
    const auto mirror_node_freedom_map_table = Kokkos::create_mirror(state.node_freedom_map_table);
    Kokkos::deep_copy(mirror_node_freedom_map_table, host_node_freedom_map_table);
    Kokkos::deep_copy(state.node_freedom_map_table, mirror_node_freedom_map_table);

    create_constraint_freedom_table(constraints, state);

    const auto host_target_node_index = create_mirror(constraints.target_node_index);
    Kokkos::deep_copy(host_target_node_index, constraints.target_node_index);
    const auto host_target_node_freedom_signature =
        create_mirror(constraints.target_node_freedom_signature);
    Kokkos::deep_copy(host_target_node_freedom_signature, constraints.target_node_freedom_signature);
    const auto host_target_node_freedom_table = create_mirror(constraints.target_node_freedom_table);
    Kokkos::deep_copy(host_target_node_freedom_table, constraints.target_node_freedom_table);

    EXPECT_EQ(host_target_node_index(0), 1);  // target node index is 1
    EXPECT_EQ(host_target_node_freedom_signature(0), FreedomSignature::AllComponents);
    for (auto k = 0U; k < 6U; ++k) {
        EXPECT_EQ(host_target_node_freedom_table(0, k), k + 6);  // DOFs: 6, 7, 8, 9, 10, 11
    }
}

}  // namespace openturbine::tests
