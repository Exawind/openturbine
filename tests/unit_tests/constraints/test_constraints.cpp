#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>

#include "constraints/constraint.hpp"
#include "constraints/constraints.hpp"

namespace openturbine::tests {

TEST(ConstraintsTest, EmptyConstructor) {
    using DeviceType = Kokkos::Device<Kokkos::DefaultExecutionSpace, Kokkos::DefaultExecutionSpace::memory_space>;
    const auto constraints = Constraints<DeviceType>(std::vector<Constraint>{}, std::vector<Node>{});
    EXPECT_EQ(constraints.num_constraints, 0);
    EXPECT_EQ(constraints.num_dofs, 0);
}

TEST(ConstraintsTest, SingleConstraintConstructorWithFixedBC) {
    auto node1 = Node(0, Array_7{0., 0., 0., 1., 0., 0., 0.});
    auto node2 = Node(1, Array_7{1., 0., 0., 1., 0., 0., 0.});
    auto constraint = Constraint(0, ConstraintType::kFixedBC, {0, 1});

    using DeviceType = Kokkos::Device<Kokkos::DefaultExecutionSpace, Kokkos::DefaultExecutionSpace::memory_space>;
    const auto constraints = Constraints<DeviceType>({constraint}, {node1, node2});
    EXPECT_EQ(constraints.num_constraints, 1);
    EXPECT_EQ(constraints.num_dofs, 6);  // Fixed constraint has 6 DOFs

    auto host_base_node_index = Kokkos::create_mirror_view(constraints.base_node_index);
    auto host_target_node_index = Kokkos::create_mirror_view(constraints.target_node_index);
    Kokkos::deep_copy(host_base_node_index, constraints.base_node_index);
    Kokkos::deep_copy(host_target_node_index, constraints.target_node_index);
    EXPECT_EQ(host_base_node_index(0), 0);
    EXPECT_EQ(host_target_node_index(0), 1);
}

TEST(ConstraintsTest, MultipleConstraintsConstructor) {
    auto node1 = Node(0, Array_7{0., 0., 0., 1., 0., 0., 0.});
    auto node2 = Node(1, Array_7{1., 0., 0., 1., 0., 0., 0.});
    auto node3 = Node(2, Array_7{2., 0., 0., 1., 0., 0., 0.});

    auto fixed_constraint = Constraint(0, ConstraintType::kFixedBC, {0, 1});

    auto revolute_constraint =
        Constraint(1, ConstraintType::kRevoluteJoint, {1, 2}, Array_3{0., 1., 0.});

    using DeviceType = Kokkos::Device<Kokkos::DefaultExecutionSpace, Kokkos::DefaultExecutionSpace::memory_space>;
    const auto constraints =
        Constraints<DeviceType>({fixed_constraint, revolute_constraint}, {node1, node2, node3});
    EXPECT_EQ(constraints.num_constraints, 2);
    EXPECT_EQ(constraints.num_dofs, 11);  // Fixed (6) + Revolute (5) = 11 DOFs
}

TEST(ConstraintsTest, UpdateDisplacementAndUpdateViews) {
    auto node1 = Node(0, Array_7{0., 0., 0., 1., 0., 0., 0.});
    auto node2 = Node(1, Array_7{1., 0., 0., 1., 0., 0., 0.});
    auto constraint = Constraint(0, ConstraintType::kFixedBC, {0, 1});

    using DeviceType = Kokkos::Device<Kokkos::DefaultExecutionSpace, Kokkos::DefaultExecutionSpace::memory_space>;
    auto constraints = Constraints<DeviceType>({constraint}, {node1, node2});

    const Array_7 new_displacement{0.1, 0.2, 0.3, 1., 0., 0., 0.};
    constraints.UpdateDisplacement(0, new_displacement);
    constraints.UpdateViews();

    auto host_input = Kokkos::create_mirror_view(constraints.input);
    Kokkos::deep_copy(host_input, constraints.input);
    for (size_t i = 0; i < 7; ++i) {
        EXPECT_DOUBLE_EQ(host_input(0, i), new_displacement[i]);
    }
}

TEST(ConstraintsTest, UpdateViewsWithControlSignal) {
    auto node1 = Node(0, Array_7{0., 0., 0., 1., 0., 0., 0.});
    double control_signal = 1.5;
    auto constraint = Constraint(
        0, ConstraintType::kRotationControl, {0, 0}, Array_3{1., 0., 0.}, &control_signal
    );

    using DeviceType = Kokkos::Device<Kokkos::DefaultExecutionSpace, Kokkos::DefaultExecutionSpace::memory_space>;
    auto constraints = Constraints<DeviceType>({constraint}, {node1});
    constraints.UpdateViews();

    auto host_input = Kokkos::create_mirror_view(constraints.input);
    Kokkos::deep_copy(host_input, constraints.input);
    EXPECT_DOUBLE_EQ(host_input(0, 0), control_signal);
}

}  // namespace openturbine::tests
