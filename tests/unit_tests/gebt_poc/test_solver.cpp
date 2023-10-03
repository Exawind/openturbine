#include <gtest/gtest.h>

#include "src/gebt_poc/solver.h"
#include "src/gen_alpha_poc/quaternion.h"
#include "tests/unit_tests/gen_alpha_poc/test_utilities.h"

namespace openturbine::gebt_poc::tests {

Kokkos::View<double*> CreateGeneralizedCoordinates(double t) {
    auto scale = 0.1;
    auto ux = scale * (t * t);
    auto uy = scale * (t * t * t - t * t);
    auto uz = scale * (t * t + 0.2 * t * t * t);
    auto quaternions =
        gen_alpha_solver::rotation_matrix_to_quaternion(gen_alpha_solver::RotationMatrix{
            1., 0., 0.,                                     // row 1
            0., std::cos(scale * t), -std::sin(scale * t),  // row 2
            0., std::sin(scale * t), std::cos(scale * t)    // row 3
        });

    Kokkos::View<double*> generalized_coords("generalized_coords", 7);
    generalized_coords(0) = ux;
    generalized_coords(1) = uy;
    generalized_coords(2) = uz;
    generalized_coords(3) = quaternions.GetScalarComponent();
    generalized_coords(4) = quaternions.GetXComponent();
    generalized_coords(5) = quaternions.GetYComponent();
    generalized_coords(6) = quaternions.GetZComponent();

    return generalized_coords;
}

TEST(SolverTest, CreateGeneralizedCoordinates) {
    {
        auto generalized_coords = CreateGeneralizedCoordinates(0.);

        EXPECT_DOUBLE_EQ(generalized_coords(0), 0.);
        EXPECT_DOUBLE_EQ(generalized_coords(1), 0.);
        EXPECT_DOUBLE_EQ(generalized_coords(2), 0.);
        // Corresponding to rotation matrix {{1.,0,0}, {0,1.,0}, {0,0,1.}}
        EXPECT_DOUBLE_EQ(generalized_coords(3), 1.);
        EXPECT_DOUBLE_EQ(generalized_coords(4), 0.);
        EXPECT_DOUBLE_EQ(generalized_coords(5), 0.);
        EXPECT_DOUBLE_EQ(generalized_coords(6), 0.);
    }

    {
        auto generalized_coords = CreateGeneralizedCoordinates(1.);

        EXPECT_DOUBLE_EQ(generalized_coords(0), 0.1);
        EXPECT_DOUBLE_EQ(generalized_coords(1), 0.);
        EXPECT_DOUBLE_EQ(generalized_coords(2), 0.12);
        // Corresponding to rotation matrix
        // {{1.,0,0}, {0,0.995004,-0.0998334}, {0,0.0998334,0.995004}}
        EXPECT_DOUBLE_EQ(generalized_coords(3), 0.99875026039496628);
        EXPECT_DOUBLE_EQ(generalized_coords(4), 0.049979169270678324);
        EXPECT_DOUBLE_EQ(generalized_coords(5), 0.);
        EXPECT_DOUBLE_EQ(generalized_coords(6), 0.);
    }

    {
        auto generalized_coords = CreateGeneralizedCoordinates(2.);

        EXPECT_DOUBLE_EQ(generalized_coords(0), 0.4);
        EXPECT_DOUBLE_EQ(generalized_coords(1), 0.4);
        EXPECT_DOUBLE_EQ(generalized_coords(2), 0.56);
        // Corresponding to rotation matrix
        // {{1.,0,0}, {0,0.980067,-0.198669}, {0,0.198669,0.980067}}
        EXPECT_DOUBLE_EQ(generalized_coords(3), 0.99500416527802571);
        EXPECT_DOUBLE_EQ(generalized_coords(4), 0.099833416646828155);
        EXPECT_DOUBLE_EQ(generalized_coords(5), 0.);
        EXPECT_DOUBLE_EQ(generalized_coords(6), 0.);
    }
}

}  // namespace openturbine::gebt_poc::tests
