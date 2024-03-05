#include <gtest/gtest.h>

#include "src/gebt_poc/NodalDynamicStiffnessMatrix.hpp"

#include "tests/unit_tests/gen_alpha_poc/test_utilities.h"

namespace openturbine::gebt_poc {
TEST(SolverTest, NodalDynamicStiffnessMatrix) {
    auto mm = gen_alpha_solver::create_matrix({
        {2.0000000000000009, 5.2041704279304213E-17, -5.5511151231257827E-17, 4.163336342344337E-17,
         0.62605214725880387, -0.33952055713492146},
        {5.2041704279304213E-17, 2.0000000000000018, 1.3877787807814457E-17, -0.62605214725880398,
         3.4694469519536142E-18, 0.22974877626536772},
        {-5.5511151231257827E-17, 1.3877787807814457E-17, 2.0000000000000013, 0.33952055713492141,
         -0.22974877626536766, 1.3877787807814457E-17},
        {-4.163336342344337E-17, -0.62605214725880387, 0.33952055713492146, 1.3196125048858467,
         1.9501108129670985, 3.5958678677753957},
        {0.62605214725880398, -3.4694469519536142E-18, -0.22974877626536772, 1.9501108129670985,
         2.881855217930184, 5.3139393458205735},
        {-0.33952055713492141, 0.22974877626536766, -1.3877787807814457E-17, 3.5958678677753957,
         5.3139393458205726, 9.7985322771839804},
    });
    auto sectional_mass_matrix = MassMatrix(mm);

    auto velocity = gen_alpha_solver::create_vector(
        {0.0025446043828620765, -0.0024798542682092665, 0.000065079641503883005,
         0.0025446043828620765, -0.0024798542682092665, 0.000065079641503883005}
    );
    auto acceleration = gen_alpha_solver::create_vector(
        {0.0025446043828620765, -0.0024151041535564553, 0.00012982975615669339,
         0.0025446043828620765, -0.0024798542682092665, -0.0024798542682092665}
    );

    auto stiffness_matrix = Kokkos::View<double**>("stiffness_matrix", 6, 6);
    NodalDynamicStiffnessMatrix(velocity, acceleration, sectional_mass_matrix, stiffness_matrix);

    openturbine::gen_alpha_solver::tests::expect_kokkos_view_2D_equal(
        stiffness_matrix,
        {// row 1
         {0., 0., 0., -0.0023904728226588536, 0.00056585276642745421, 0.0005703830914904407},
         // row 2
         {0., 0., 0., -0.00085994394592263162, -0.00097181181209263397, 0.00084261536265676736},
         // row 3
         {0., 0., 0., -0.0015972403418206974, 0.0015555222717217175, -0.00025743506367869402},
         // row 4
         {0., 0., 0., 0.0047622883054215057, -0.016524233223710137, 0.007213755243428677},
         // row 5
         {0., 0., 0., 0.035164381478288514, 0.017626317482204206, -0.022463736936512112},
         // row 6
         {0., 0., 0., -0.0025828596476940593, 0.042782118352914907, -0.022253736971069835}}
    );
}
}