#include <gtest/gtest.h>

#include "src/elements/springs/create_springs.hpp"
#include "src/elements/springs/springs.hpp"
#include "src/model/model.hpp"
#include "src/step/assemble_stiffness_matrix_springs.hpp"

namespace openturbine::tests {

inline auto SetUpSpringStiffnessTest() {
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

TEST(AssembleStiffnessMatrixSpringsTest, StiffnessMatrixMatchesUnitStiffnessInXDirection) {
    auto springs = SetUpSpringStiffnessTest();

    auto a_host = Kokkos::create_mirror_view(springs.a);
    a_host(0, 0, 0) = 1.;  // Only X-direction stiffness
    Kokkos::deep_copy(springs.a, a_host);

    AssembleStiffnessMatrixSprings(springs);
    auto stiffness_host = Kokkos::create_mirror_view(springs.stiffness_matrix_terms);
    Kokkos::deep_copy(stiffness_host, springs.stiffness_matrix_terms);

    for (size_t i = 0; i < springs.num_elems; ++i) {
        // Node 1-1 block
        EXPECT_DOUBLE_EQ(stiffness_host(i, 0, 0, 0, 0), 1.);
        EXPECT_DOUBLE_EQ(stiffness_host(i, 0, 0, 0, 1), 0.);
        EXPECT_DOUBLE_EQ(stiffness_host(i, 0, 0, 0, 2), 0.);
        EXPECT_DOUBLE_EQ(stiffness_host(i, 0, 0, 1, 0), 0.);
        EXPECT_DOUBLE_EQ(stiffness_host(i, 0, 0, 1, 1), 0.);
        EXPECT_DOUBLE_EQ(stiffness_host(i, 0, 0, 1, 2), 0.);
        EXPECT_DOUBLE_EQ(stiffness_host(i, 0, 0, 2, 0), 0.);
        EXPECT_DOUBLE_EQ(stiffness_host(i, 0, 0, 2, 1), 0.);
        EXPECT_DOUBLE_EQ(stiffness_host(i, 0, 0, 2, 2), 0.);

        // Node 1-2 block
        EXPECT_DOUBLE_EQ(stiffness_host(i, 0, 1, 0, 0), -1.);
        EXPECT_DOUBLE_EQ(stiffness_host(i, 0, 1, 0, 1), 0.);
        EXPECT_DOUBLE_EQ(stiffness_host(i, 0, 1, 0, 2), 0.);
        EXPECT_DOUBLE_EQ(stiffness_host(i, 0, 1, 1, 0), 0.);
        EXPECT_DOUBLE_EQ(stiffness_host(i, 0, 1, 1, 1), 0.);
        EXPECT_DOUBLE_EQ(stiffness_host(i, 0, 1, 1, 2), 0.);
        EXPECT_DOUBLE_EQ(stiffness_host(i, 0, 1, 2, 0), 0.);
        EXPECT_DOUBLE_EQ(stiffness_host(i, 0, 1, 2, 1), 0.);
        EXPECT_DOUBLE_EQ(stiffness_host(i, 0, 1, 2, 2), 0.);

        // Node 2-1 block
        EXPECT_DOUBLE_EQ(stiffness_host(i, 1, 0, 0, 0), -1.);
        EXPECT_DOUBLE_EQ(stiffness_host(i, 1, 0, 0, 1), 0.);
        EXPECT_DOUBLE_EQ(stiffness_host(i, 1, 0, 0, 2), 0.);
        EXPECT_DOUBLE_EQ(stiffness_host(i, 1, 0, 1, 0), 0.);
        EXPECT_DOUBLE_EQ(stiffness_host(i, 1, 0, 1, 1), 0.);
        EXPECT_DOUBLE_EQ(stiffness_host(i, 1, 0, 1, 2), 0.);
        EXPECT_DOUBLE_EQ(stiffness_host(i, 1, 0, 2, 0), 0.);
        EXPECT_DOUBLE_EQ(stiffness_host(i, 1, 0, 2, 1), 0.);
        EXPECT_DOUBLE_EQ(stiffness_host(i, 1, 0, 2, 2), 0.);

        // Node 2-2 block
        EXPECT_DOUBLE_EQ(stiffness_host(i, 1, 1, 0, 0), 1.);
        EXPECT_DOUBLE_EQ(stiffness_host(i, 1, 1, 0, 1), 0.);
        EXPECT_DOUBLE_EQ(stiffness_host(i, 1, 1, 0, 2), 0.);
        EXPECT_DOUBLE_EQ(stiffness_host(i, 1, 1, 1, 0), 0.);
        EXPECT_DOUBLE_EQ(stiffness_host(i, 1, 1, 1, 1), 0.);
        EXPECT_DOUBLE_EQ(stiffness_host(i, 1, 1, 1, 2), 0.);
        EXPECT_DOUBLE_EQ(stiffness_host(i, 1, 1, 2, 0), 0.);
        EXPECT_DOUBLE_EQ(stiffness_host(i, 1, 1, 2, 1), 0.);
        EXPECT_DOUBLE_EQ(stiffness_host(i, 1, 1, 2, 2), 0.);
    }
}

TEST(AssembleStiffnessMatrixSpringsTest, StiffnessMatrixMatchesGeneralStiffness) {
    auto springs = SetUpSpringStiffnessTest();

    auto a_host = Kokkos::create_mirror_view(springs.a);
    // Set all components to create the following stiffness matrix for springs.a
    // | 1 2 3 |
    // | 2 4 6 |
    // | 3 6 9 |
    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            a_host(0, i, j) = static_cast<double>((i + 1) * (j + 1));
        }
    }
    Kokkos::deep_copy(springs.a, a_host);

    AssembleStiffnessMatrixSprings(springs);
    auto stiffness_host = Kokkos::create_mirror_view(springs.stiffness_matrix_terms);
    Kokkos::deep_copy(stiffness_host, springs.stiffness_matrix_terms);

    for (size_t i = 0; i < springs.num_elems; ++i) {
        for (size_t m = 0; m < 3; ++m) {
            for (size_t n = 0; n < 3; ++n) {
                auto value = static_cast<double>((m + 1) * (n + 1));
                // Node 1-1 block
                EXPECT_DOUBLE_EQ(stiffness_host(i, 0, 0, m, n), value);
                // Node 1-2 block
                EXPECT_DOUBLE_EQ(stiffness_host(i, 0, 1, m, n), -value);
                // Node 2-1 block
                EXPECT_DOUBLE_EQ(stiffness_host(i, 1, 0, m, n), -value);
                // Node 2-2 block
                EXPECT_DOUBLE_EQ(stiffness_host(i, 1, 1, m, n), value);
            }
        }
    }
}

}  // namespace openturbine::tests
