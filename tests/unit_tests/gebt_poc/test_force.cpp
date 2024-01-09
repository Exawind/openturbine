#include <gtest/gtest.h>

#include "src/gebt_poc/force.h"
#include "tests/unit_tests/gen_alpha_poc/test_utilities.h"

namespace openturbine::gebt_poc::tests {

TEST(GeneralizedForcesTest, CreateGeneralizedForcesWithDefaultValues) {
    auto generalized_forces = GeneralizedForces();

    gen_alpha_solver::tests::expect_kokkos_view_1D_equal(
        generalized_forces.GetGeneralizedForces(), {0., 0., 0., 0., 0., 0.}
    );
    EXPECT_EQ(generalized_forces.GetNode(), 1);
}

TEST(GeneralizedForcesTest, CreateGeneralizedForcesWithGivenValues) {
    auto forces = gen_alpha_solver::Vector{1., 2., 3.};
    auto moments = gen_alpha_solver::Vector{4., 5., 6.};
    auto node = 2;
    auto generalized_forces = GeneralizedForces(forces, moments, node);

    gen_alpha_solver::tests::expect_kokkos_view_1D_equal(
        generalized_forces.GetGeneralizedForces(), {1., 2., 3., 4., 5., 6.}
    );
    EXPECT_EQ(generalized_forces.GetNode(), 2);
}

TEST(GeneralizedForcesTest, CreateGeneralizedForcesWithGiven1DVector) {
    auto generalized_forces =
        GeneralizedForces(gen_alpha_solver::create_vector({1., 2., 3., 4., 5., 6.}), 5);

    gen_alpha_solver::tests::expect_kokkos_view_1D_equal(
        generalized_forces.GetGeneralizedForces(), {1., 2., 3., 4., 5., 6.}
    );

    auto f = generalized_forces.GetForces();
    EXPECT_NEAR(f.GetXComponent(), 1., 1.e-15);
    EXPECT_NEAR(f.GetYComponent(), 2., 1.e-15);
    EXPECT_NEAR(f.GetZComponent(), 3., 1.e-15);

    auto m = generalized_forces.GetMoments();
    EXPECT_NEAR(m.GetXComponent(), 4., 1.e-15);
    EXPECT_NEAR(m.GetYComponent(), 5., 1.e-15);
    EXPECT_NEAR(m.GetZComponent(), 6., 1.e-15);

    EXPECT_EQ(generalized_forces.GetNode(), 5);
}

TEST(TimeVaryingForcesTest, CreateTimeVaryingForcesWithGivenFunction) {
    auto create_time_varying_force = [](double time) {
        return gen_alpha_solver::create_vector(
            {1. + time, 2. + time, 3. + time, 4. + time, 5. + time, 6. + time}
        );
    };
    auto node = 1;
    auto time_varying_forces = TimeVaryingForces(create_time_varying_force, node);

    gen_alpha_solver::tests::expect_kokkos_view_1D_equal(
        time_varying_forces.GetGeneralizedForces(0.), {1., 2., 3., 4., 5., 6.}
    );
    gen_alpha_solver::tests::expect_kokkos_view_1D_equal(
        time_varying_forces.GetGeneralizedForces(1.), {2., 3., 4., 5., 6., 7.}
    );
    EXPECT_EQ(time_varying_forces.GetNode(), 1);
}

}  // namespace openturbine::gebt_poc::tests
