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
    auto invalid_node = Node(0U, {0., 0., 0., 1., 0., 0., 0.});  // base node - index is 0
    auto node_1 = Node(4U, {1., 0., 0., 1., 0., 0., 0.});        // target node - index is 4
    auto prescribed_bc =
        std::make_shared<Constraint>(ConstraintType::kPrescribedBC, 1, invalid_node, node_1);
    auto constraints = Constraints(std::vector<std::shared_ptr<Constraint>>{prescribed_bc});

    auto state = State(5U);  // 5 nodes in the system
    constexpr auto host_node_freedom_map_table_data = std::array{0UL, 6UL, 15UL, 18UL, 24UL};
    const auto host_node_freedom_map_table = Kokkos::View<size_t[5], Kokkos::HostSpace>::const_type(
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

    EXPECT_EQ(host_target_node_index(0), 4);  // target node index is 4
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
    auto node_1 = Node(1U, {0., 0., 0., 1., 0., 0., 0.});  // base node - index is 1
    auto node_2 = Node(2U, {1., 0., 0., 1., 0., 0., 0.});  // target node - index is 2
    auto rigid_bc = std::make_shared<Constraint>(ConstraintType::kRigidJoint, 1, node_1, node_2);
    auto constraints = Constraints(std::vector<std::shared_ptr<Constraint>>{rigid_bc});

    auto state = State(3U);  // 3 nodes in the system
    constexpr auto host_node_freedom_map_table_data = std::array{0UL, 3UL, 9UL};
    const auto host_node_freedom_map_table = Kokkos::View<size_t[3], Kokkos::HostSpace>::const_type(
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

    EXPECT_EQ(host_base_node_index(0), 1);    // base node index is 1
    EXPECT_EQ(host_target_node_index(0), 2);  // target node index is 2
    EXPECT_EQ(
        host_base_node_freedom_signature(0), FreedomSignature::AllComponents
    );  // base node has 6 DOFs
    EXPECT_EQ(
        host_target_node_freedom_signature(0), FreedomSignature::AllComponents
    );  // target node has 6 DOFs which will be fixed relative to base node
    for (auto k = 0U; k < 6U; ++k) {
        EXPECT_EQ(host_base_node_freedom_table(0, k), k + 3);  // base node DOFs: 3, 4, 5, 6, 7, 8
    }
    for (auto k = 0U; k < 6U; ++k) {
        EXPECT_EQ(
            host_target_node_freedom_table(0, k), k + 9
        );  // target node DOFs: 9, 10, 11, 12, 13, 14
    }
}

TEST(TestCreateConstraintFreedomTable, DoubeNodeConstraint_RevoluteJoint) {
    auto node_1 = Node(3U, {0., 0., 0., 1., 0., 0., 0.});   // base node - index is 3
    auto node_2 = Node(11U, {1., 0., 0., 1., 0., 0., 0.});  // target node - index is 11
    const Array_3 rotation_axis = {0., 0., 1.};
    double torque = 0.;
    auto revolute_joint = std::make_shared<Constraint>(
        ConstraintType::kRevoluteJoint, 1, node_1, node_2, rotation_axis, &torque
    );
    auto constraints = Constraints(std::vector<std::shared_ptr<Constraint>>{revolute_joint});

    auto state = State(12U);  // 12 nodes in the system
    constexpr auto host_node_freedom_map_table_data =
        std::array{0UL, 3UL, 6UL, 9UL, 12UL, 15UL, 18UL, 21UL, 24UL, 27UL, 30UL, 33UL};
    const auto host_node_freedom_map_table = Kokkos::View<size_t[12], Kokkos::HostSpace>::const_type(
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

    EXPECT_EQ(host_base_node_index(0), 3);     // base node index is 3
    EXPECT_EQ(host_target_node_index(0), 11);  // target node index is 11
    EXPECT_EQ(
        host_base_node_freedom_signature(0), FreedomSignature::AllComponents
    );  // base node has all 6 DOFs
    EXPECT_EQ(
        host_target_node_freedom_signature(0), FreedomSignature::AllComponents
    );  // target node has all 6 DOFs as well

    for (auto k = 0U; k < 6U; ++k) {
        EXPECT_EQ(
            host_base_node_freedom_table(0, k), k + 9
        );  // base node DOFs: 9, 10, 11, 12, 13, 14
    }
    for (auto k = 0U; k < 6U; ++k) {
        EXPECT_EQ(
            host_target_node_freedom_table(0, k), k + 33
        );  // target node DOFs: 33, 34, 35, 36, 37, 38
    }
}

}  // namespace openturbine::tests
