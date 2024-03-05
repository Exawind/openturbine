#include <gtest/gtest.h>

#include "src/gebt_poc/NodalInertialForces.hpp"
#include "tests/unit_tests/gen_alpha_poc/test_utilities.h"

namespace openturbine::gebt_poc {

TEST(SolverTest, NodalInertialForces) {
    auto mm = gen_alpha_solver::create_matrix({
        {2., 0., 0., 0., 0.601016, -0.398472},                   // row 1
        {0., 2., -1.45094e-19, -0.601016, -5.78775e-20, 0.2},    // row 2
        {0., -1.45094e-19, 2., 0.398472, -0.2, -7.22267e-20},    // row 3
        {0., -0.601016, 0.398472, 1., 1.99236, 3.00508},         // row 4
        {0.601016, 5.78775e-20, -0.2, 1.99236, 3.9695, 5.9872},  // row 5
        {-0.398472, 0.2, 7.22267e-20, 3.00508, 5.9872, 9.0305}   // row 6
    });
    auto sectional_mass_matrix = MassMatrix(mm);

    auto velocity = gen_alpha_solver::create_vector(
        {0.0025446, -0.00247985, 0.0000650796, 0.0025446, -0.00247985, 0.0000650796}
    );
    auto acceleration = gen_alpha_solver::create_vector(
        {0.0025446, -0.0024151, 0.00012983, 0.0025446, -0.00247985, -0.00247985}
    );

    auto inertial_forces_fc = Kokkos::View<double[6]>("inertial_forces_fc");
    NodalInertialForces(velocity, acceleration, sectional_mass_matrix, inertial_forces_fc);

    openturbine::gen_alpha_solver::tests::expect_kokkos_view_1D_equal(
        inertial_forces_fc,
        {0.00458328, -0.00685947, 0.00176196, -0.00832838, -0.0181013, -0.0311086}
    );
}

}  // namespace openturbine::gebt_poc