/** \file test_heat_solve.cpp
 *
 *  Tests the methods in the heat solver module
 */

#include "src/heat_solve.H"
#include "src/utilities/debug_utils.H"

#include <vector>
#include <numeric>
#include "gtest/gtest.h"

namespace oturb_tests {

TEST(HeatSolve_linspace, argtypes)
{
    // Being a template function, linspace should accept any numeric argument
    // for start, stop and step
    // Successfully compiling and linking means the test passes
    // std::vector<double> test_lin1 = openturbine::heat_solve::linspace(1, 2, 3);
    std::vector<double> test_lin2 = openturbine::heat_solve::linspace(1.0, 2.0, 3);
    // std::vector<double> test_lin3 = openturbine::heat_solve::linspace(1, 2.0, 3);

    // this should handle negative arguments just the same
    std::vector<double> test_lin4 = openturbine::heat_solve::linspace(-2.0, 0.0, 3);
    std::vector<double> truth_vec4{ -2.0, -1.0, 0.0 };
    ASSERT_EQ( truth_vec4, test_lin4 );
}

TEST(HeatSolve_linspace, error_handling)
{
    ASSERT_THROW(
        openturbine::heat_solve::linspace(1.0, -1.0, -3),
        std::invalid_argument
    );
}

TEST(HeatSolve_linspace, values)
{
    std::vector<double> truth_vec;

    // linspace should equally distribute points including at the beginning and end
    // Test the trivial case of 3 points: 0.0, 1.0, 2.0
    truth_vec = std::vector{ 0.0, 1.0, 2.0 };
    ASSERT_EQ(
        openturbine::heat_solve::linspace(0.0, 2.0, 3),
        truth_vec
    );

    // It should go in the opposite direction if needed
    truth_vec = std::vector{ 2.0, 1.0, 0.0 };
    ASSERT_EQ(
        openturbine::heat_solve::linspace(2.0, 0.0, 3),
        truth_vec
    );

    // 0 step size should yield a 0-size vector
    std::vector<double> zero_lin = openturbine::heat_solve::linspace(0.0, 2.0, 0);
    ASSERT_EQ( zero_lin.size(), 0 );

    // 1 step size should yield a 1-size vector with the starting point
    std::vector<double> one_lin = openturbine::heat_solve::linspace(0.0, 2.0, 1);
    ASSERT_EQ( one_lin.size(), 1 );
    ASSERT_EQ( one_lin[0], 0.0 );
}

TEST(HeatSolve, FullSolve)
{
    int axis_size = 11;                // Size of the 1D grid
    double side_length = 1.0;          // Length of the 1D domain
    double k = 1.011;                  // Thermal diffusivity of the material in units of cm^2/s

    // For all points at 0.0, the domain should be all 0.0 after a few iterations.
    // This tests for any numerical noise.
    auto U = static_cast<double*>(std::malloc(axis_size * sizeof(double)));
    U = openturbine::heat_solve::heat_conduction_solver(
        axis_size,
        side_length,
        5,     // number of iterations
        k,
        0.0,   // x=0 IC
        0.0,   // x=end IC
        1e-5   // tolerance
    );
    ASSERT_EQ( std::accumulate(U, U + axis_size, 0.0), 0.0 );
    std::free(U);

    // For equal IC, the full domain should neraly converage to the same value.
    // Note the lowered tolerance here. The middle points will only be exact
    // after the iterative error approaches machine precision, so we expect
    // a small amount of difference.
    double test_temp = 10.0;
    U = static_cast<double*>(std::malloc(axis_size * sizeof(double)));
    U = openturbine::heat_solve::heat_conduction_solver(
        axis_size,
        side_length,
        500,   // number of iterations
        k,
        test_temp,  // x=0 IC
        test_temp,  // x=end IC
        1e-8   // tolerance
    );
    ASSERT_NEAR( std::accumulate(U, U + axis_size, 0.0), test_temp * axis_size / side_length, 1e-6 );

    // This heat model is actually just a line. For an odd number of grid points,
    // the middle point should have a value equal to half the difference of the
    // end points.
    test_temp = 10.0;
    U = static_cast<double*>(std::malloc(axis_size * sizeof(double)));
    U = openturbine::heat_solve::heat_conduction_solver(
        axis_size,
        side_length,
        500,   // number of iterations
        k,
        0.0,   // x=0 IC
        test_temp,  // x=end IC
        1e-8   // tolerance
    );
    ASSERT_NEAR( U[5], test_temp / 2.0, 1e-6 );
}


} // namespace oturb_tests
