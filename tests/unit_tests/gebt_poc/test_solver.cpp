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

Kokkos::View<double**> AssignGeneralizedCoordinatesToNodes(std::size_t order) {
    auto nodes = GenerateGLLPoints(order);
    auto generalized_coords = Kokkos::View<double**>("generalized_coords", order + 1, 7);
    for (std::size_t i = 0; i < order + 1; ++i) {
        auto xi = (nodes[i] + 1.) / 2.;
        auto gen_coords = CreateGeneralizedCoordinates(xi);
        Kokkos::parallel_for(
            Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, 7),
            KOKKOS_LAMBDA(const int j) { generalized_coords(i, j) = gen_coords(j); }
        );
    }
    return generalized_coords;
}

TEST(SolverTest, AssignGeneralizedCoordinatesToNodes) {
    auto generalized_coords = AssignGeneralizedCoordinatesToNodes(4);

    // Node 1
    EXPECT_DOUBLE_EQ(generalized_coords(0, 0), 0.);
    EXPECT_DOUBLE_EQ(generalized_coords(0, 1), 0.);
    EXPECT_DOUBLE_EQ(generalized_coords(0, 2), 0.);
    // Corresponding to rotation matrix {{1.,0,0}, {0,1.,0}, {0,0,1.}}
    EXPECT_DOUBLE_EQ(generalized_coords(0, 3), 1.);
    EXPECT_DOUBLE_EQ(generalized_coords(0, 4), 0.);
    EXPECT_DOUBLE_EQ(generalized_coords(0, 5), 0.);
    EXPECT_DOUBLE_EQ(generalized_coords(0, 6), 0.);

    // Node 2
    EXPECT_DOUBLE_EQ(generalized_coords(1, 0), 0.0029816021788868566);
    EXPECT_DOUBLE_EQ(generalized_coords(1, 1), -0.00246675949494302);
    EXPECT_DOUBLE_EQ(generalized_coords(1, 2), 0.0030845707156756234);
    // Corresponding to rotation matrix
    // {{1.,0,0}, {0,0.999851,-0.0172665}, {0,0.0172665,0.999851}}
    EXPECT_DOUBLE_EQ(generalized_coords(1, 3), 0.99996273020427251);
    EXPECT_DOUBLE_EQ(generalized_coords(1, 4), 0.008633550973807835);
    EXPECT_DOUBLE_EQ(generalized_coords(1, 5), 0.);
    EXPECT_DOUBLE_EQ(generalized_coords(1, 6), 0.);

    // Node 3
    EXPECT_DOUBLE_EQ(generalized_coords(2, 0), 0.025);
    EXPECT_DOUBLE_EQ(generalized_coords(2, 1), -0.0125);
    EXPECT_DOUBLE_EQ(generalized_coords(2, 2), 0.0275);
    // Corresponding to rotation matrix
    // {{1.,0,0}, {0,0.99875,-0.0499792}, {0,0.0499792,0.99875}}
    EXPECT_DOUBLE_EQ(generalized_coords(2, 3), 0.99968751627570251);
    EXPECT_DOUBLE_EQ(generalized_coords(2, 4), 0.024997395914712332);
    EXPECT_DOUBLE_EQ(generalized_coords(2, 5), 0.);
    EXPECT_DOUBLE_EQ(generalized_coords(2, 6), 0.);

    // Node 4
    EXPECT_DOUBLE_EQ(generalized_coords(3, 0), 0.068446969249684589);
    EXPECT_DOUBLE_EQ(generalized_coords(3, 1), -0.011818954790771262);
    EXPECT_DOUBLE_EQ(generalized_coords(3, 2), 0.079772572141467255);
    // Corresponding to rotation matrix
    // {{1.,0,0},{0,0.99658,-0.0826383},{0,0.0826383,0.99658}
    EXPECT_DOUBLE_EQ(generalized_coords(3, 3), 0.99914453488230548);
    EXPECT_DOUBLE_EQ(generalized_coords(3, 4), 0.041354545274025191);
    EXPECT_DOUBLE_EQ(generalized_coords(3, 5), 0.);
    EXPECT_DOUBLE_EQ(generalized_coords(3, 6), 0.);

    // Node 5
    EXPECT_DOUBLE_EQ(generalized_coords(4, 0), 0.1);
    EXPECT_DOUBLE_EQ(generalized_coords(4, 1), 0.);
    EXPECT_DOUBLE_EQ(generalized_coords(4, 2), 0.12);
    // Corresponding to rotation matrix
    // {{1.,0,0}, {0,0.995004,-0.0998334}, {0,0.0998334,0.995004}}
    EXPECT_DOUBLE_EQ(generalized_coords(4, 3), 0.99875026039496628);
    EXPECT_DOUBLE_EQ(generalized_coords(4, 4), 0.049979169270678324);
    EXPECT_DOUBLE_EQ(generalized_coords(4, 5), 0.);
    EXPECT_DOUBLE_EQ(generalized_coords(4, 6), 0.);
}

}  // namespace openturbine::gebt_poc::tests
