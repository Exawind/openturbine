#include <gtest/gtest.h>

#include "src/elements/springs/create_springs.hpp"
#include "src/elements/springs/springs.hpp"
#include "src/model/model.hpp"
#include "src/step/assemble_residual_vector_springs.hpp"

namespace openturbine::tests {

inline auto SetUpSpringResidualTest() {
    auto model = Model();
    model.AddNode(
        {0, 0, 0, 1, 0, 0, 0}, {0, 0, 0, 1, 0, 0, 0}, {0, 0, 0, 0, 0, 0}, {0, 0, 0, 0, 0, 0}
    );
    model.AddNode(
        {1, 0, 0, 1, 0, 0, 0}, {0, 0, 0, 1, 0, 0, 0}, {0, 0, 0, 0, 0, 0}, {0, 0, 0, 0, 0, 0}
    );
    const auto springs_input =
        SpringsInput({SpringElement(std::array{model.GetNode(0), model.GetNode(1)}, 1., 1.)});
    return CreateSprings(springs_input);
}

TEST(AssembleResidualVectorSpringsTest, ResidualVectorIsZeroWhenNoForcesApplied) {
    auto springs = SetUpSpringResidualTest();
    // springs.f is zero
    AssembleResidualVectorSprings(springs);
    auto residual_host = Kokkos::create_mirror_view(springs.residual_vector_terms);
    Kokkos::deep_copy(residual_host, springs.residual_vector_terms);

    for (size_t i = 0; i < springs.num_elems; ++i) {
        EXPECT_DOUBLE_EQ(residual_host(i, 0, 0), 0.);  // Node 1 DOF 1
        EXPECT_DOUBLE_EQ(residual_host(i, 0, 1), 0.);  // Node 1 DOF 2
        EXPECT_DOUBLE_EQ(residual_host(i, 0, 2), 0.);  // Node 1 DOF 3
        EXPECT_DOUBLE_EQ(residual_host(i, 1, 0), 0.);  // Node 2 DOF 1
        EXPECT_DOUBLE_EQ(residual_host(i, 1, 1), 0.);  // Node 2 DOF 2
        EXPECT_DOUBLE_EQ(residual_host(i, 1, 2), 0.);  // Node 2 DOF 3
    }
}

TEST(AssembleResidualVectorSpringsTest, ResidualVectorMatchesUnitForceInXDirection) {
    auto springs = SetUpSpringResidualTest();
    auto f_host = Kokkos::create_mirror_view(springs.f);
    f_host(0, 0) = 1.;   // Node 1 DOF 1
    Kokkos::deep_copy(springs.f, f_host);

    AssembleResidualVectorSprings(springs);
    auto residual_host = Kokkos::create_mirror_view(springs.residual_vector_terms);
    Kokkos::deep_copy(residual_host, springs.residual_vector_terms);

    for (size_t i = 0; i < springs.num_elems; ++i) {
        EXPECT_DOUBLE_EQ(residual_host(i, 0, 0), 1.);   // Node 1 DOF 1
        EXPECT_DOUBLE_EQ(residual_host(i, 0, 1), 0.);   // Node 1 DOF 2
        EXPECT_DOUBLE_EQ(residual_host(i, 0, 2), 0.);   // Node 1 DOF 3
        EXPECT_DOUBLE_EQ(residual_host(i, 1, 0), -1.);  // Node 2 DOF 1
        EXPECT_DOUBLE_EQ(residual_host(i, 1, 1), 0.);   // Node 2 DOF 2
        EXPECT_DOUBLE_EQ(residual_host(i, 1, 2), 0.);   // Node 2 DOF 3
    }
}

TEST(AssembleResidualVectorSpringsTest, ResidualVectorMatchesAppliedForcesInAllDirections) {
    auto springs = SetUpSpringResidualTest();

    auto f_host = Kokkos::create_mirror_view(springs.f);
    f_host(0, 0) = 2.;    // Node 1 DOF 1
    f_host(0, 1) = 1.5;   // Node 1 DOF 2
    f_host(0, 2) = -1.;   // Node 1 DOF 3
    Kokkos::deep_copy(springs.f, f_host);

    AssembleResidualVectorSprings(springs);
    auto residual_host = Kokkos::create_mirror_view(springs.residual_vector_terms);
    Kokkos::deep_copy(residual_host, springs.residual_vector_terms);

    for (size_t i = 0; i < springs.num_elems; ++i) {
        EXPECT_DOUBLE_EQ(residual_host(i, 0, 0), 2.);    // Node 1 DOF 1
        EXPECT_DOUBLE_EQ(residual_host(i, 0, 1), 1.5);   // Node 1 DOF 2
        EXPECT_DOUBLE_EQ(residual_host(i, 0, 2), -1.);   // Node 1 DOF 3
        EXPECT_DOUBLE_EQ(residual_host(i, 1, 0), -2.);   // Node 2 DOF 1
        EXPECT_DOUBLE_EQ(residual_host(i, 1, 1), -1.5);  // Node 2 DOF 2
        EXPECT_DOUBLE_EQ(residual_host(i, 1, 2), 1.);    // Node 2 DOF 3
    }
}

}  // namespace openturbine::tests
