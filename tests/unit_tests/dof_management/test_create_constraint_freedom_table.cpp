#include <gtest/gtest.h>

#include "src/dof_management/assemble_node_freedom_allocation_table.hpp"
#include "src/dof_management/compute_node_freedom_map_table.hpp"
#include "src/dof_management/create_constraint_freedom_table.hpp"
#include "src/model/model.hpp"

namespace openturbine::tests {

TEST(TestCreateConstraintFreedomTable, SingleNodeConstraint_FixedBC) {
    auto invalid_node = Node(0U, Array_7{0., 0., 0., 1., 0., 0., 0.});  // base node - index is 0
    auto node_1 = Node(1U, Array_7{1., 0., 0., 1., 0., 0., 0.});        // target node - index is 1
    auto fixed_bc = Constraint(0, ConstraintType::kFixedBC, {0, 1});
    auto constraints = Constraints({fixed_bc}, {invalid_node, node_1});

    auto elements = Elements();
    auto state = State(2U);  // 2 nodes in the system
    assemble_node_freedom_allocation_table(state, elements, constraints);

    constexpr auto host_node_freedom_map_table_data = std::array{0UL, 17UL};
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
    EXPECT_EQ(
        host_target_node_freedom_signature(0), FreedomSignature::AllComponents
    );  // taget node has 6 DOFs which will be fixed
    for (auto k = 0U; k < 6U; ++k) {
        EXPECT_EQ(
            host_target_node_freedom_table(0, k), k + 17
        );  // target node DOFs: 17, 18, 19, 20, 21, 22
    }
}

TEST(TestCreateConstraintFreedomTable, SingleNodeConstraint_PrescribedBC) {
    auto invalid_node = Node(0U, Array_7{0., 0., 0., 1., 0., 0., 0.});  // base node - index is 0
    auto node_1 = Node(1U, Array_7{1., 0., 0., 1., 0., 0., 0.});        // target node - index is 1
    auto prescribed_bc = Constraint(0, ConstraintType::kPrescribedBC, {0, 1});
    auto constraints = Constraints({prescribed_bc}, {invalid_node, node_1});

    auto elements = Elements();
    auto state = State(2U);  // 2 nodes in the system
    assemble_node_freedom_allocation_table(state, elements, constraints);

    constexpr auto host_node_freedom_map_table_data = std::array{0UL, 24UL};
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

    EXPECT_EQ(host_target_node_index(0), 1);  // target node index is 4
    EXPECT_EQ(
        host_target_node_freedom_signature(0), FreedomSignature::AllComponents
    );  // taget node has 6 DOFs which will be prescribed
    for (auto k = 0U; k < 6U; ++k) {
        EXPECT_EQ(
            host_target_node_freedom_table(0, k), k + 24
        );  // target node DOFs: 24, 25, 26, 27, 28, 29
    }
}

TEST(TestCreateConstraintFreedomTable, DoubeNodeConstraint_RigidBC) {
    auto node_1 = Node(0U, Array_7{0., 0., 0., 1., 0., 0., 0.});  // base node - index is 0
    auto node_2 = Node(1U, Array_7{1., 0., 0., 1., 0., 0., 0.});  // target node - index is 1
    auto rigid_bc = Constraint(0, ConstraintType::kRigidJoint, {0, 1});
    auto constraints = Constraints({rigid_bc}, {node_1, node_2});

    auto elements = Elements();
    auto state = State(2U);  // 2 nodes in the system
    assemble_node_freedom_allocation_table(state, elements, constraints);

    constexpr auto host_node_freedom_map_table_data = std::array{0UL, 6UL};
    const auto host_node_freedom_map_table = Kokkos::View<size_t[2], Kokkos::HostSpace>::const_type(
        host_node_freedom_map_table_data.data()
    );
    const auto mirror_node_freedom_map_table = Kokkos::create_mirror(state.node_freedom_map_table);
    Kokkos::deep_copy(mirror_node_freedom_map_table, host_node_freedom_map_table);
    Kokkos::deep_copy(state.node_freedom_map_table, mirror_node_freedom_map_table);

    create_constraint_freedom_table(constraints, state);

    const auto host_base_node_index = create_mirror(constraints.base_node_index);
    Kokkos::deep_copy(host_base_node_index, constraints.base_node_index);
    const auto host_target_node_index = create_mirror(constraints.target_node_index);
    Kokkos::deep_copy(host_target_node_index, constraints.target_node_index);
    const auto host_base_node_freedom_signature =
        create_mirror(constraints.base_node_freedom_signature);
    Kokkos::deep_copy(host_base_node_freedom_signature, constraints.base_node_freedom_signature);
    const auto host_target_node_freedom_signature =
        create_mirror(constraints.target_node_freedom_signature);
    Kokkos::deep_copy(host_target_node_freedom_signature, constraints.target_node_freedom_signature);
    const auto host_base_node_freedom_table = create_mirror(constraints.base_node_freedom_table);
    Kokkos::deep_copy(host_base_node_freedom_table, constraints.base_node_freedom_table);
    const auto host_target_node_freedom_table = create_mirror(constraints.target_node_freedom_table);
    Kokkos::deep_copy(host_target_node_freedom_table, constraints.target_node_freedom_table);

    EXPECT_EQ(host_base_node_index(0), 0);    // base node index is 0
    EXPECT_EQ(host_target_node_index(0), 1);  // target node index is 1
    EXPECT_EQ(
        host_base_node_freedom_signature(0), FreedomSignature::AllComponents
    );  // base node has 6 DOFs
    EXPECT_EQ(
        host_target_node_freedom_signature(0), FreedomSignature::AllComponents
    );  // target node has 6 DOFs which will be fixed relative to base node
    for (auto k = 0U; k < 6U; ++k) {
        EXPECT_EQ(host_base_node_freedom_table(0, k), k);  // base node DOFs: 0, 1, 2, 3, 4, 5
    }
    for (auto k = 0U; k < 6U; ++k) {
        EXPECT_EQ(
            host_target_node_freedom_table(0, k), k + 6
        );  // target node DOFs: 6, 7, 8, 9, 10, 11
    }
}

}  // namespace openturbine::tests