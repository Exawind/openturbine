#include <gtest/gtest.h>

#include "src/gebt_poc/ElementalStaticForcesResidual.hpp"
#include "tests/unit_tests/gebt_poc/test_data.h"
#include "tests/unit_tests/gen_alpha_poc/test_utilities.h"

namespace openturbine::gebt_poc {

TEST(SolverTest, ElementalStaticForcesResidualWithZeroValues) {
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
        {1., 0., 0., 0., 0., 0.},  // row 1
        {0., 1., 0., 0., 0., 0.},  // row 2
        {0., 0., 1., 0., 0., 0.},  // row 3
        {0., 0., 0., 1., 0., 0.},  // row 4
        {0., 0., 0., 0., 1., 0.},  // row 5
        {0., 0., 0., 0., 0., 1.}   // row 6
    });

    // 7-point Gauss-Legendre quadrature
    auto quadrature = CreateGaussLegendreQuadrature(7);

    auto residual = Kokkos::View<double[30]>("residual");
    ElementalStaticForcesResidual(
        position_vectors, generalized_coords, stiffness, quadrature, residual
    );

    openturbine::gen_alpha_solver::tests::expect_kokkos_view_1D_equal(
        residual,
        // residuals expected to be all zeros
        {0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
         0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.}
    );
}

struct NonZeroValues_populate_position {
    Kokkos::View<double[5][7]> position_vectors;

    KOKKOS_FUNCTION
    void operator()(std::size_t) const {
        // node 1
        position_vectors(0, 0) = 0.;
        position_vectors(0, 1) = 0.;
        position_vectors(0, 2) = 0.;
        position_vectors(0, 3) = 0.9778215200524469;
        position_vectors(0, 4) = -0.01733607539094763;
        position_vectors(0, 5) = -0.09001900002195001;
        position_vectors(0, 6) = -0.18831121859148398;
        // node 2
        position_vectors(1, 0) = 0.8633658232300573;
        position_vectors(1, 1) = -0.25589826392541715;
        position_vectors(1, 2) = 0.1130411210682743;
        position_vectors(1, 3) = 0.9950113028068008;
        position_vectors(1, 4) = -0.002883848832932071;
        position_vectors(1, 5) = -0.030192109815745303;
        position_vectors(1, 6) = -0.09504013471947484;
        // node 3
        position_vectors(2, 0) = 2.5;
        position_vectors(2, 1) = -0.25;
        position_vectors(2, 2) = 0.;
        position_vectors(2, 3) = 0.9904718430204884;
        position_vectors(2, 4) = -0.009526411091536478;
        position_vectors(2, 5) = 0.09620741150793366;
        position_vectors(2, 6) = 0.09807604012323785;
        // node 4
        position_vectors(3, 0) = 4.136634176769943;
        position_vectors(3, 1) = 0.39875540678255983;
        position_vectors(3, 2) = -0.5416125496397027;
        position_vectors(3, 3) = 0.9472312341234699;
        position_vectors(3, 4) = -0.049692141629315074;
        position_vectors(3, 5) = 0.18127630174800594;
        position_vectors(3, 6) = 0.25965858850765167;
        // node 5
        position_vectors(4, 0) = 5.;
        position_vectors(4, 1) = 1.;
        position_vectors(4, 2) = -1.;
        position_vectors(4, 3) = 0.9210746582719719;
        position_vectors(4, 4) = -0.07193653093139739;
        position_vectors(4, 5) = 0.20507529985516368;
        position_vectors(4, 6) = 0.32309554437664584;
    }
};

struct NonZeroValues_populate_coords {
    Kokkos::View<double[5][7]> generalized_coords;

    KOKKOS_FUNCTION
    void operator()(std::size_t) const {
        // node 1
        generalized_coords(0, 0) = 0.;
        generalized_coords(0, 1) = 0.;
        generalized_coords(0, 2) = 0.;
        generalized_coords(0, 3) = 1.;
        generalized_coords(0, 4) = 0.;
        generalized_coords(0, 5) = 0.;
        generalized_coords(0, 6) = 0.;
        // node 2
        generalized_coords(1, 0) = 0.0029816021788868583;
        generalized_coords(1, 1) = -0.0024667594949430213;
        generalized_coords(1, 2) = 0.0030845707156756256;
        generalized_coords(1, 3) = 0.9999627302042724;
        generalized_coords(1, 4) = 0.008633550973807838;
        generalized_coords(1, 5) = 0.;
        generalized_coords(1, 6) = 0.;
        // node 3
        generalized_coords(2, 0) = 0.025;
        generalized_coords(2, 1) = -0.0125;
        generalized_coords(2, 2) = 0.027500000000000004;
        generalized_coords(2, 3) = 0.9996875162757026;
        generalized_coords(2, 4) = 0.024997395914712332;
        generalized_coords(2, 5) = 0.;
        generalized_coords(2, 6) = 0.;
        // node 4
        generalized_coords(3, 0) = 0.06844696924968456;
        generalized_coords(3, 1) = -0.011818954790771264;
        generalized_coords(3, 2) = 0.07977257214146723;
        generalized_coords(3, 3) = 0.9991445348823056;
        generalized_coords(3, 4) = 0.04135454527402519;
        generalized_coords(3, 5) = 0.;
        generalized_coords(3, 6) = 0.;
        // node 5
        generalized_coords(4, 0) = 0.1;
        generalized_coords(4, 1) = 0.;
        generalized_coords(4, 2) = 0.12;
        generalized_coords(4, 3) = 0.9987502603949662;
        generalized_coords(4, 4) = 0.049979169270678324;
        generalized_coords(4, 5) = 0.;
        generalized_coords(4, 6) = 0.;
    }
};

TEST(SolverTest, ElementalStaticForcesResidualWithNonZeroValues) {
    auto position_vectors = Kokkos::View<double[5][7]>("position_vectors");
    Kokkos::parallel_for(1, NonZeroValues_populate_position{position_vectors});

    auto generalized_coords = Kokkos::View<double[5][7]>("generalized_coords");
    Kokkos::parallel_for(1, NonZeroValues_populate_coords{generalized_coords});

    auto stiffness = gen_alpha_solver::create_matrix({
        {1., 2., 3., 4., 5., 6.},       // row 1
        {2., 4., 6., 8., 10., 12.},     // row 2
        {3., 6., 9., 12., 15., 18.},    // row 3
        {4., 8., 12., 16., 20., 24.},   // row 4
        {5., 10., 15., 20., 25., 30.},  // row 5
        {6., 12., 18., 24., 30., 36.}   // row 6
    });

    auto quadrature = CreateGaussLegendreQuadrature(7);

    auto residual = Kokkos::View<double[30]>("residual");

    ElementalStaticForcesResidual(
        position_vectors, generalized_coords, stiffness, quadrature, residual
    );

    openturbine::gen_alpha_solver::tests::expect_kokkos_view_1D_equal(
        residual, tests::expected_residual
    );
}
}  // namespace openturbine::gebt_poc