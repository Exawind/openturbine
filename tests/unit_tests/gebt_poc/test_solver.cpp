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

Kokkos::View<double*> CreatePositionVectors(double t) {
    auto rx = 5. * t;
    auto ry = -2. * t + 3. * t * t;
    auto rz = t - 2. * (t * t);

    Kokkos::View<double*> position_vector("position_vector", 3);
    position_vector(0) = rx;
    position_vector(1) = ry;
    position_vector(2) = rz;

    return position_vector;
}

TEST(SolverTest, CreatePositionVectors) {
    // Use linear shape functions to interpolate the position vector
    auto pt_1 = LagrangePolynomial(1, -1.);
    auto pt_2 = LagrangePolynomial(1, -0.6546536707079771);
    auto pt_3 = LagrangePolynomial(1, 0.);
    auto pt_4 = LagrangePolynomial(1, 0.6546536707079771);
    auto pt_5 = LagrangePolynomial(1, 1.);

    {
        auto position_vector = CreatePositionVectors(pt_1[1]);

        EXPECT_DOUBLE_EQ(position_vector(0), 0.);
        EXPECT_DOUBLE_EQ(position_vector(1), 0.);
        EXPECT_DOUBLE_EQ(position_vector(2), 0.);
    }

    {
        auto position_vector = CreatePositionVectors(pt_2[1]);

        EXPECT_DOUBLE_EQ(position_vector(0), 0.86336582323005728);
        EXPECT_DOUBLE_EQ(position_vector(1), -0.25589826392541715);
        EXPECT_DOUBLE_EQ(position_vector(2), 0.1130411210682743);
    }

    {
        auto position_vector = CreatePositionVectors(pt_3[1]);

        EXPECT_DOUBLE_EQ(position_vector(0), 2.5);
        EXPECT_DOUBLE_EQ(position_vector(1), -0.25);
        EXPECT_DOUBLE_EQ(position_vector(2), 0.);
    }

    {
        auto position_vector = CreatePositionVectors(pt_4[1]);

        EXPECT_DOUBLE_EQ(position_vector(0), 4.1366341767699426);
        EXPECT_DOUBLE_EQ(position_vector(1), 0.39875540678255983);
        EXPECT_DOUBLE_EQ(position_vector(2), -0.54161254963970273);
    }

    {
        auto position_vector = CreatePositionVectors(pt_5[1]);

        EXPECT_DOUBLE_EQ(position_vector(0), 5.);
        EXPECT_DOUBLE_EQ(position_vector(1), 1.);
        EXPECT_DOUBLE_EQ(position_vector(2), -1.);
    }
}

TEST(SolverTest, UserDefinedQuadrature) {
    auto quadrature_points = std::vector<double>{
        -0.9491079123427585,  // point 1
        -0.7415311855993945,  // point 2
        -0.4058451513773972,  // point 3
        0.,                   // point 4
        0.4058451513773972,   // point 5
        0.7415311855993945,   // point 6
        0.9491079123427585    // point 7
    };
    auto quadrature_weights = std::vector<double>{
        0.1294849661688697,  // weight 1
        0.2797053914892766,  // weight 2
        0.3818300505051189,  // weight 3
        0.4179591836734694,  // weight 4
        0.3818300505051189,  // weight 5
        0.2797053914892766,  // weight 6
        0.1294849661688697   // weight 7
    };
    auto quadrature_rule = UserDefinedQuadrature(quadrature_points, quadrature_weights);

    EXPECT_EQ(quadrature_rule.GetNumQuadraturePoints(), 7);
    EXPECT_EQ(quadrature_rule.GetQuadraturePoints(), quadrature_points);
    EXPECT_EQ(quadrature_rule.GetQuadratureWeights(), quadrature_weights);
}

}  // namespace openturbine::gebt_poc::tests
