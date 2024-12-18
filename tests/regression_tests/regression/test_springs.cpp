#include <gtest/gtest.h>

#include "test_utilities.hpp"

#include "src/elements/springs/create_springs.hpp"
#include "src/elements/springs/springs.hpp"
#include "src/model/model.hpp"

namespace openturbine::tests {

inline auto SetUpSprings() {
    auto model = Model();
    // Add two nodes for the spring element
    model.AddNode(
        {0, 0, 0, 1, 0, 0, 0},  // First node at origin
        {0, 0, 0, 1, 0, 0, 0}, {0, 0, 0, 0, 0, 0}, {0, 0, 0, 0, 0, 0}
    );
    model.AddNode(
        {1, 0, 0, 1, 0, 0, 0},  // Second node at (1,0,0)
        {0, 0, 0, 1, 0, 0, 0}, {0, 0, 0, 0, 0, 0}, {0, 0, 0, 0, 0, 0}
    );

    const auto springs_input = SpringsInput({SpringElement(
        std::array{model.GetNode(0), model.GetNode(1)},
        1000.,  // Spring stiffness
        1.      // Undeformed length
    )});

    return CreateSprings(springs_input);
}

TEST(SpringsTest, NodeStateIndices) {
    const auto springs = SetUpSprings();
    auto node_state_indices_host = Kokkos::create_mirror(springs.node_state_indices);
    Kokkos::deep_copy(node_state_indices_host, springs.node_state_indices);
    EXPECT_EQ(node_state_indices_host(0, 0), 0);
    EXPECT_EQ(node_state_indices_host(0, 1), 1);
}

TEST(SpringsTest, InitialPositionVector) {
    const auto springs = SetUpSprings();
    expect_kokkos_view_2D_equal(
        springs.x0,
        {
            {1., 0., 0.},  // Vector from node 0 to node 1
        }
    );
}

TEST(SpringsTest, ReferenceLength) {
    const auto springs = SetUpSprings();
    expect_kokkos_view_1D_equal(springs.l_ref, {1.});  // Undeformed length
}

TEST(SpringsTest, SpringStiffness) {
    const auto springs = SetUpSprings();
    expect_kokkos_view_1D_equal(springs.k, {1000.});  // Spring stiffness
}

}  // namespace openturbine::tests
