#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>

#include "src/constraints/constraint.hpp"
#include "src/constraints/constraints.hpp"

namespace openturbine::tests {

TEST(ConstraintsTest, EmptyConstructor) {
    const auto constraints = Constraints(std::vector<std::shared_ptr<Constraint>>{});
    EXPECT_EQ(constraints.num_constraints, 0);
    EXPECT_EQ(constraints.num_dofs, 0);
}

TEST(ConstraintsTest, SingleConstraintConstructor) {
    auto node1 = Node(0, Array_7{0., 0., 0., 1., 0., 0., 0.});
    auto node2 = Node(1, Array_7{1., 0., 0., 1., 0., 0., 0.});
    auto constraint = std::make_shared<Constraint>(ConstraintType::kFixedBC, 0, node1, node2);

    const auto constraints = Constraints({constraint});
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

    auto fixed_constraint = std::make_shared<Constraint>(ConstraintType::kFixedBC, 0, node1, node2);

    auto revolute_constraint = std::make_shared<Constraint>(
        ConstraintType::kRevoluteJoint, 1, node2, node3, Array_3{0., 1., 0.}
    );

    const auto constraints = Constraints({fixed_constraint, revolute_constraint});
    EXPECT_EQ(constraints.num_constraints, 2);
    EXPECT_EQ(constraints.num_dofs, 11);  // Fixed (6) + Revolute (5) = 11 DOFs
}

TEST(ConstraintsTest, UpdateDisplacementAndUpdateViews) {
    auto node1 = Node(0, Array_7{0., 0., 0., 1., 0., 0., 0.});  // invalid node
    auto node2 = Node(1, Array_7{1., 0., 0., 1., 0., 0., 0.});  // target node
    auto constraint = std::make_shared<Constraint>(ConstraintType::kFixedBC, 0, node1, node2);

    auto constraints = Constraints({constraint});

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
    auto constraint = std::make_shared<Constraint>(
        ConstraintType::kRotationControl, 0, node1, node1, Array_3{1., 0., 0.}, &control_signal
    );

    auto constraints = Constraints({constraint});
    constraints.UpdateViews();

    auto host_input = Kokkos::create_mirror_view(constraints.input);
    Kokkos::deep_copy(host_input, constraints.input);
    EXPECT_DOUBLE_EQ(host_input(0, 0), control_signal);
}

}  // namespace openturbine::tests