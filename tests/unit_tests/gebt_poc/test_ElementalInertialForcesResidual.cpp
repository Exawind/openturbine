#include <gtest/gtest.h>

#include "src/gebt_poc/ElementalInertialForcesResidual.hpp"
#include "tests/unit_tests/gen_alpha_poc/test_utilities.h"

namespace openturbine::gebt_poc {
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

struct NonZeroValues_PopulateVelocity {
    Kokkos::View<double[5][6]> velocity;

    KOKKOS_FUNCTION
    void operator()(std::size_t) const {
        // node 1
        velocity(0, 0) = 0.;
        velocity(0, 1) = 0.;
        velocity(0, 2) = 0.;
        velocity(0, 3) = 0.;
        velocity(0, 4) = 0.;
        velocity(0, 5) = 0.;
        // node 2
        velocity(1, 0) = 0.017267316464601147;
        velocity(1, 1) = -0.014285714285714289;
        velocity(1, 2) = 0.0030845707156756256;
        velocity(1, 3) = 0.017267316464601147;
        velocity(1, 4) = -0.014285714285714289;
        velocity(1, 5) = 0.0030845707156756256;
        // node 3
        velocity(2, 0) = 0.05;
        velocity(2, 1) = -0.025;
        velocity(2, 2) = 0.0275;
        velocity(2, 3) = 0.05;
        velocity(2, 4) = -0.025;
        velocity(2, 5) = 0.0275;
        // node 4
        velocity(3, 0) = 0.08273268353539887;
        velocity(3, 1) = -0.01428571428571429;
        velocity(3, 2) = 0.07977257214146723;
        velocity(3, 3) = 0.08273268353539887;
        velocity(3, 4) = -0.01428571428571429;
        velocity(3, 5) = 0.07977257214146723;
        // node 5
        velocity(4, 0) = 0.1;
        velocity(4, 1) = 0.;
        velocity(4, 2) = 0.12;
        velocity(4, 3) = 0.1;
        velocity(4, 4) = 0.;
        velocity(4, 5) = 0.12;
    }
};

struct NonZeroValues_PopulateAcceleration {
    Kokkos::View<double[5][6]> acceleration;

    KOKKOS_FUNCTION
    void operator()(std::size_t) const {
        // node 1
        acceleration(0, 0) = 0.;
        acceleration(0, 1) = 0.;
        acceleration(0, 2) = 0.;
        acceleration(0, 3) = 0.;
        acceleration(0, 4) = 0.;
        acceleration(0, 5) = 0.;
        // node 2
        acceleration(1, 0) = 0.017267316464601147;
        acceleration(1, 1) = -0.01130411210682743;
        acceleration(1, 2) = 0.006066172894562484;
        acceleration(1, 3) = 0.017267316464601147;
        acceleration(1, 4) = -0.014285714285714289;
        acceleration(1, 5) = -0.014285714285714289;
        // node 3
        acceleration(2, 0) = 0.05;
        acceleration(2, 1) = 0.;
        acceleration(2, 2) = 0.0525;
        acceleration(2, 3) = 0.05;
        acceleration(2, 4) = -0.025;
        acceleration(2, 5) = -0.025;
        // node 4
        acceleration(3, 0) = 0.08273268353539887;
        acceleration(3, 1) = 0.05416125496397028;
        acceleration(3, 2) = 0.1482195413911518;
        acceleration(3, 3) = 0.08273268353539887;
        acceleration(3, 4) = -0.01428571428571429;
        acceleration(3, 5) = -0.01428571428571429;
        // node 5
        acceleration(4, 0) = 0.1;
        acceleration(4, 1) = 0.1;
        acceleration(4, 2) = 0.22;
        acceleration(4, 3) = 0.1;
        acceleration(4, 4) = 0.;
        acceleration(4, 5) = 0.;
    }
};

TEST(SolverTest, ElementalInertialForcesResidualWithNonZeroValues) {
    auto position_vectors = Kokkos::View<double[5][7]>("position_vectors");
    Kokkos::parallel_for(1, NonZeroValues_populate_position{position_vectors});

    auto generalized_coords = Kokkos::View<double[5][7]>("generalized_coords");
    Kokkos::parallel_for(1, NonZeroValues_populate_coords{generalized_coords});

    auto quadrature_points =
        std::vector{-0.9491079123427585, -0.7415311855993945, -0.4058451513773972, 0.,
                    0.4058451513773972,  0.7415311855993945,  0.9491079123427585};
    auto quadrature_weights =
        std::vector{0.1294849661688697, 0.2797053914892766, 0.3818300505051189, 0.4179591836734694,
                    0.3818300505051189, 0.2797053914892766, 0.1294849661688697};
    auto quadrature = UserDefinedQuadrature(quadrature_points, quadrature_weights);

    auto velocity = Kokkos::View<double[5][6]>("velocity");
    Kokkos::parallel_for(1, NonZeroValues_PopulateVelocity{velocity});

    auto acceleration = Kokkos::View<double[5][6]>("acceleration");
    Kokkos::parallel_for(1, NonZeroValues_PopulateAcceleration{acceleration});

    auto sectional_mass_matrix = gen_alpha_solver::create_matrix({
        {2., 0., 0., 0., 0.6, -0.4},  // row 1
        {0., 2., 0., -0.6, 0., 0.2},  // row 2
        {0., 0., 2., 0.4, -0.2, 0.},  // row 3
        {0., -0.6, 0.4, 1., 2., 3.},  // row 4
        {0.6, 0., -0.2, 2., 4., 6.},  // row 5
        {-0.4, 0.2, 0., 3., 6., 9.}   // row 6
    });

    auto residual = Kokkos::View<double[30]>("residual");

    ElementalInertialForcesResidual(
        position_vectors, generalized_coords, velocity, acceleration, sectional_mass_matrix,
        quadrature, residual
    );

    openturbine::gen_alpha_solver::tests::expect_kokkos_view_1D_equal(
        residual, {0.00011604556408927589, -0.0006507362696177887, -0.0006134866787567209,
                   0.0006142322011935119,  -0.002199479688149071,  -0.0024868433546726618,
                   0.04224129527348785,    -0.04970288500953028,   0.03088887284431359,
                   -0.06975512417597271,   -0.1016119340697192,    -0.21457597550060847,
                   0.17821954823792258,    -0.06852557905980934,   0.23918323280829631,
                   -0.11742114482709062,   -0.25775479677677565,   -0.3851496964119588,
                   0.2842963634125304,     0.09461848004333043,    0.5753721231363035,
                   -0.03722226802421836,   -0.08186055756075582,   -0.023972312767030792,
                   0.07251940986370664,    0.047582193710461386,   0.1727148583550359,
                   -0.007667261048492517,  0.02731289347020833,    0.05581821163081137}
    );
}

}  // namespace openturbine::gebt_poc