#include <gtest/gtest.h>

#include "src/gebt_poc/SectionalMassMatrix.hpp"
#include "tests/unit_tests/gen_alpha_poc/test_utilities.h"

namespace openturbine::gebt_poc {

TEST(SolverTest, SectionalMassMatrix) {
    auto rotation_0 = gen_alpha_solver::create_matrix({
        {0.92468736109510075, 0.34700636042507571, -0.156652066872805},
        {-0.34223371517230472, 0.93786259974465924, 0.05735702397746531},
        {0.16682136682793711, 0.00057430369330956077, 0.98598696834437271},
    });
    auto rotation = gen_alpha_solver::create_matrix({
        {1.0000000000000002, 0, 0},
        {0, 0.99999676249603286, -0.0025446016295712901},
        {0, 0.0025446016295712901, 0.99999676249603286},
    });

    auto Mass = gen_alpha_solver::create_matrix({
        {2., 0., 0., 0., 0.6, -0.4},  // row 1
        {0., 2., 0., -0.6, 0., 0.2},  // row 2
        {0., 0., 2., 0.4, -0.2, 0.},  // row 3
        {0., -0.6, 0.4, 1., 2., 3.},  // row 4
        {0.6, 0., -0.2, 2., 4., 6.},  // row 5
        {-0.4, 0.2, 0., 3., 6., 9.},  // row 6
    });

    auto sectional_mass = Kokkos::View<double[6][6]>("sectional_mass");
    SectionalMassMatrix(Mass, rotation_0, rotation, sectional_mass);

    openturbine::gen_alpha_solver::tests::expect_kokkos_view_2D_equal(
        sectional_mass,
        {
            {2.0000000000000009, 5.2041704279304213E-17, -5.5511151231257827E-17,
             4.163336342344337E-17, 0.62605214725880387, -0.33952055713492146},
            {5.2041704279304213E-17, 2.0000000000000018, 1.3877787807814457E-17,
             -0.62605214725880398, 3.4694469519536142E-18, 0.22974877626536772},
            {-5.5511151231257827E-17, 1.3877787807814457E-17, 2.0000000000000013,
             0.33952055713492141, -0.22974877626536766, 1.3877787807814457E-17},
            {-4.163336342344337E-17, -0.62605214725880387, 0.33952055713492146, 1.3196125048858467,
             1.9501108129670985, 3.5958678677753957},
            {0.62605214725880398, -3.4694469519536142E-18, -0.22974877626536772, 1.9501108129670985,
             2.881855217930184, 5.3139393458205735},
            {-0.33952055713492141, 0.22974877626536766, -1.3877787807814457E-17, 3.5958678677753957,
             5.3139393458205726, 9.7985322771839804},
        }
    );
}

}  // namespace openturbine::gebt_poc