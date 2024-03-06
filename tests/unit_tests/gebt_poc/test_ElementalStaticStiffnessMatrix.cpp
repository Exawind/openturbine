#include <gtest/gtest.h>

#include "src/gebt_poc/ElementalStaticStiffnessMatrix.hpp"
#include "tests/unit_tests/gebt_poc/test_data.h"
#include "tests/unit_tests/gen_alpha_poc/test_utilities.h"

namespace openturbine::gebt_poc {

TEST(SolverTest, ElementalStaticStiffnessMatrixWithZeroValues) {
    // 5 nodes on a line
    auto position_vectors = gen_alpha_solver::create_matrix({
        {0., 0., 0., 1., 0., 0., 0.},  // node 1
        {1., 0., 0., 1., 0., 0., 0.},  // node 2
        {2., 0., 0., 1., 0., 0., 0.},  // node 3
        {3., 0., 0., 1., 0., 0., 0.},  // node 4
        {4., 0., 0., 1., 0., 0., 0.}   // node 5
    });

    // zero displacement and rotation
    auto generalized_coords = gen_alpha_solver::create_matrix({
        {0., 0., 0., 0., 0., 0., 0.},  // node 1
        {0., 0., 0., 0., 0., 0., 0.},  // node 2
        {0., 0., 0., 0., 0., 0., 0.},  // node 3
        {0., 0., 0., 0., 0., 0., 0.},  // node 4
        {0., 0., 0., 0., 0., 0., 0.}   // node 5
    });

    // identity stiffness matrix
    auto stiffness = gen_alpha_solver::create_matrix({
        {0., 0., 0., 0., 0., 0.},  // row 1
        {0., 0., 0., 0., 0., 0.},  // row 2
        {0., 0., 0., 0., 0., 0.},  // row 3
        {0., 0., 0., 0., 0., 0.},  // row 4
        {0., 0., 0., 0., 0., 0.},  // row 5
        {0., 0., 0., 0., 0., 0.}   // row 6
    });

    // 7-point Gauss-Legendre quadrature
    auto quadrature = UserDefinedQuadrature(
        {-0.9491079123427585, -0.7415311855993945, -0.4058451513773972, 0., 0.4058451513773972,
         0.7415311855993945, 0.9491079123427585},
        {0.1294849661688697, 0.2797053914892766, 0.3818300505051189, 0.4179591836734694,
         0.3818300505051189, 0.2797053914892766, 0.1294849661688697}
    );

    auto iteration_matrix = Kokkos::View<double[30][30]>("iteration_matrix");
    ElementalStaticStiffnessMatrix(
        position_vectors, generalized_coords, stiffness, quadrature,
        iteration_matrix
    );

    openturbine::gen_alpha_solver::tests::expect_kokkos_view_2D_equal(
        iteration_matrix, tests::zeros_30x30
    );
}

TEST(SolverTest, ElementalStaticStiffnessMatrixWithNonZeroValues2D) {
    auto position_vectors = gen_alpha_solver::create_matrix(
        {// node 1
         {0., 0., 0., 0.9778215200524469, -0.01733607539094763, -0.09001900002195001,
          -0.18831121859148398},
         // node 2
         {0.8633658232300573, -0.25589826392541715, 0.1130411210682743, 0.9950113028068008,
          -0.002883848832932071, -0.030192109815745303, -0.09504013471947484},
         // node 3
         {2.5, -0.25, 0., 0.9904718430204884, -0.009526411091536478, 0.09620741150793366,
          0.09807604012323785},
         // node 4
         {4.136634176769943, 0.39875540678255983, -0.5416125496397027, 0.9472312341234699,
          -0.049692141629315074, 0.18127630174800594, 0.25965858850765167},
         // node 5
         {5., 1., -1., 0.9210746582719719, -0.07193653093139739, 0.20507529985516368,
          0.32309554437664584}}
    );

    auto generalized_coords = gen_alpha_solver::create_matrix(
        {// node 1
         {0., 0., 0., 1., 0., 0., 0.},
         // node 2
         {0.0029816021788868583, -0.0024667594949430213, 0.0030845707156756256, 0.9999627302042724,
          0.008633550973807838, 0., 0.},
         // node 3
         {0.025, -0.0125, 0.027500000000000004, 0.9996875162757026, 0.024997395914712332, 0., 0.},
         // node 4
         {0.06844696924968456, -0.011818954790771264, 0.07977257214146723, 0.9991445348823056,
          0.04135454527402519, 0., 0.},
         // node 5
         {0.1, 0., 0.12, 0.9987502603949662, 0.049979169270678324, 0., 0.}}
    );

    auto stiffness = gen_alpha_solver::create_matrix({
        {1., 2., 3., 4., 5., 6.},       // row 1
        {2., 4., 6., 8., 10., 12.},     // row 2
        {3., 6., 9., 12., 15., 18.},    // row 3
        {4., 8., 12., 16., 20., 24.},   // row 4
        {5., 10., 15., 20., 25., 30.},  // row 5
        {6., 12., 18., 24., 30., 36.}   // row 6
    });

    auto quadrature_points =
               std::vector{-0.9491079123427585, -0.7415311855993945, -0.4058451513773972, 0.,
                            0.4058451513773972,  0.7415311855993945,  0.9491079123427585};
    auto quadrature_weights = std::vector{
        0.1294849661688697, 0.2797053914892766, 0.3818300505051189, 0.4179591836734694,
        0.3818300505051189, 0.2797053914892766, 0.1294849661688697};
    auto quadrature = UserDefinedQuadrature(quadrature_points, quadrature_weights);

    auto iteration_matrix = Kokkos::View<double[30][30]>("iteration_matrix");
    ElementalStaticStiffnessMatrix(
        position_vectors, generalized_coords, stiffness, quadrature, iteration_matrix
    );

    openturbine::gen_alpha_solver::tests::expect_kokkos_view_2D_equal(
        iteration_matrix, tests::expected_iteration_matrix
    );
}

}  // namespace openturbine::gebt_poc