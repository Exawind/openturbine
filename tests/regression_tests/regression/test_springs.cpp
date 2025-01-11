#include <gtest/gtest.h>

#include "test_utilities.hpp"

#include "src/elements/springs/create_springs.hpp"
#include "src/elements/springs/springs.hpp"
#include "src/model/model.hpp"
#include "src/step/update_system_variables_springs.hpp"

namespace openturbine::tests {

inline auto SetUpSprings() {
    auto model = Model();

    // Add two nodes for the spring element
    auto node1_id =
        model.AddNode().SetPosition(0., 0., 0., 1., 0., 0., 0.).Build();  // First node at origin
    auto node2_id =
        model.AddNode().SetPosition(1., 0., 0., 1., 0., 0., 0.).Build();  // Second node at (1,0,0)

    const auto k = 10.;  // Spring stiffness coeff.
    const auto l0 = 1.;  // Undeformed length
    model.AddSpringElement(node1_id, node2_id, k, l0);

    auto springs = model.CreateSprings();
    auto state = model.CreateState();

    return std::make_tuple(springs, state);
}

TEST(SpringsTest, NodeStateIndices) {
    auto [springs, _] = SetUpSprings();
    auto node_state_indices_host = Kokkos::create_mirror(springs.node_state_indices);
    Kokkos::deep_copy(node_state_indices_host, springs.node_state_indices);
    EXPECT_EQ(node_state_indices_host(0, 0), 0);
    EXPECT_EQ(node_state_indices_host(0, 1), 1);
}

TEST(SpringsTest, InitialPositionVector) {
    auto [springs, _] = SetUpSprings();
    expect_kokkos_view_2D_equal(
        springs.x0,
        {
            {1., 0., 0.},  // Vector from node 0 to node 1
        }
    );
}

TEST(SpringsTest, ReferenceLength) {
    auto [springs, _] = SetUpSprings();
    expect_kokkos_view_1D_equal(springs.l_ref, {1.});  // Undeformed length
}

TEST(SpringsTest, SpringStiffness) {
    auto [springs, _] = SetUpSprings();
    expect_kokkos_view_1D_equal(springs.k, {10.});  // Spring stiffness
}

TEST(SpringsTest, SpringsForceWithZeroDisplacement) {
    auto [springs, state] = SetUpSprings();
    UpdateSystemVariablesSprings(springs, state);

    auto l_ref_host = Kokkos::create_mirror(springs.l_ref);
    auto l_host = Kokkos::create_mirror(springs.l);
    auto c1_host = Kokkos::create_mirror(springs.c1);

    Kokkos::deep_copy(l_ref_host, springs.l_ref);
    Kokkos::deep_copy(l_host, springs.l);
    Kokkos::deep_copy(c1_host, springs.c1);

    EXPECT_DOUBLE_EQ(l_ref_host(0), 1.);  // Undeformed length = 1.
    EXPECT_DOUBLE_EQ(l_host(0), 1.);      // Current length = 1.
    EXPECT_DOUBLE_EQ(c1_host(0), 0.);     // c1 = 0.
    expect_kokkos_view_1D_equal(
        Kokkos::subview(springs.f, 0, Kokkos::ALL), {0., 0., 0.}
    );  // No force
    expect_kokkos_view_2D_equal(
        Kokkos::subview(springs.a, 0, Kokkos::ALL, Kokkos::ALL),
        {{-10., 0., 0.}, {0., 0., 0.}, {0., 0., 0.}}
    );
}

TEST(SpringsTest, SpringsForceWithUnitDisplacement) {
    auto [springs, state] = SetUpSprings();

    auto q = Kokkos::create_mirror(state.q);
    q(1, 0) = 1.;  // Displace second node in x direction
    Kokkos::deep_copy(state.q, q);

    UpdateSystemVariablesSprings(springs, state);

    auto l_ref_host = Kokkos::create_mirror(springs.l_ref);
    auto l_host = Kokkos::create_mirror(springs.l);
    auto c1_host = Kokkos::create_mirror(springs.c1);
    auto c2_host = Kokkos::create_mirror(springs.c2);

    Kokkos::deep_copy(l_ref_host, springs.l_ref);
    Kokkos::deep_copy(l_host, springs.l);
    Kokkos::deep_copy(c1_host, springs.c1);
    Kokkos::deep_copy(c2_host, springs.c2);

    EXPECT_DOUBLE_EQ(l_ref_host(0), 1.);                       // Undeformed length = 1.
    EXPECT_DOUBLE_EQ(l_host(0), 2.);                           // Current length = 2.
    EXPECT_DOUBLE_EQ(c1_host(0), -5.);                         // c1 = -5.
    EXPECT_DOUBLE_EQ(c2_host(0), 10. * 1. / std::pow(2., 3));  // c2 = 10. / 8.
    expect_kokkos_view_1D_equal(
        Kokkos::subview(springs.f, 0, Kokkos::ALL), {-10., 0., 0.}
    );  // Force = c1 * {r}
    expect_kokkos_view_2D_equal(
        Kokkos::subview(springs.a, 0, Kokkos::ALL, Kokkos::ALL),
        {{-10., 0., 0.}, {0., -5., 0.}, {0., 0., -5.}}
    );
}

}  // namespace openturbine::tests
