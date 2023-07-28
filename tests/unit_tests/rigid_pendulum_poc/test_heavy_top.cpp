#include <gtest/gtest.h>

#include "src/rigid_pendulum_poc/generalized_alpha_time_integrator.h"
#include "src/rigid_pendulum_poc/heavy_top.h"
#include "tests/unit_tests/rigid_pendulum_poc/test_utilities.h"

namespace openturbine::rigid_pendulum::tests {

TEST(HeavyTopProblemFromBrulsAndCardona2010PaperTest, CalculateResidualVector) {
    auto M = MassMatrix(15., Vector(0.234375, 0.46875, 0.234375));
    auto mass_matrix = M.GetMassMatrix();
    auto rotation_matrix = create_matrix(
        {{0.617251, -0.757955, 0.210962},
         {0.775967, 0.63076, -0.00416521},
         {-0.129909, 0.166271, 0.977485}}
    );
    auto acceleration_vector = create_vector({1., 1., 1., 1., 1., 1.});
    auto gen_forces_vector = create_vector({0., 0., 147.150000, -0.234375, 0., 0.234375});
    auto position_vector = create_vector({0., 1., 0.});
    auto lagrange_multipliers = create_vector({1., 2., 3.});

    auto residual_vector = heavy_top_residual_vector(
        mass_matrix, rotation_matrix, acceleration_vector, gen_forces_vector, position_vector,
        lagrange_multipliers
    );

    expect_kokkos_view_1D_equal(
        residual_vector,
        {
            14.,         // row 1
            13.,         // row 2
            159.150000,  // row 3
            3.135087,    // row 4
            0.46875,     // row 5
            -1.310708    // row 6
        }
    );
}

TEST(HeavyTopProblemFromBrulsAndCardona2010PaperTest, CalculateTangentDampingMatrix) {
    auto angular_velocity_vector = create_vector({0.3, 0.1, 0.8});

    auto mass = 15.;
    auto mass_matrix = MassMatrix(mass, Vector(0.234375, 0.46875, 0.234375));
    auto inertia_matrix = mass_matrix.GetMomentOfInertiaMatrix();

    auto tangent_damping_matrix =
        heavy_top_tangent_damping_matrix(angular_velocity_vector, inertia_matrix);

    expect_kokkos_view_2D_equal(
        tangent_damping_matrix,
        {
            {0., 0., 0., 0., 0., 0.},               // row 1
            {0., 0., 0., 0., 0., 0.},               // row 2
            {0., 0., 0., 0., 0., 0.},               // row 3
            {0., 0., 0., 0., -0.1875, -0.0234375},  // row 4
            {0., 0., 0., 0., 0., 0.},               // row 5
            {0., 0., 0., 0.0234375, 0.0703125, 0.}  // row 6
        }
    );
}

TEST(HeavyTopProblemFromBrulsAndCardona2010PaperTest, CalculateTangentStiffnessMatrix) {
    auto position_vector = create_vector({0., 1., 0.});
    auto rotation_matrix = create_matrix(
        {{0.617251, -0.757955, 0.210962},
         {0.775967, 0.63076, -0.00416521},
         {-0.129909, 0.166271, 0.977485}}
    );
    auto lagrange_multipliers = create_vector({1., 2., 3.});

    auto tangent_stiffness_matrix =
        heavy_top_tangent_stiffness_matrix(position_vector, rotation_matrix, lagrange_multipliers);

    expect_kokkos_view_2D_equal(
        tangent_stiffness_matrix,
        {
            {0., 0., 0., 0., 0., 0.},               // row 1
            {0., 0., 0., 0., 0., 0.},               // row 2
            {0., 0., 0., 0., 0., 0.},               // row 3
            {0., 0., 0., -1.002378, 1.779458, 0.},  // row 4
            {0., 0., 0., 0., 0., 0.},               // row 5
            {0., 0., 0., 0., 3.135086, -1.002378}   // row 6
        }
    );
}

TEST(HeavyTopProblemFromBrulsAndCardona2010PaperTest, CalculateConstraintGradientMatrix) {
    auto position_vector = create_vector({0., 1., 0.});
    auto rotation_matrix = create_matrix(
        {{0.617251, -0.757955, 0.210962},
         {0.775967, 0.63076, -0.00416521},
         {-0.129909, 0.166271, 0.977485}}
    );

    auto constraint_gradient_matrix =
        heavy_top_constraint_gradient_matrix(position_vector, rotation_matrix);

    expect_kokkos_view_2D_equal(
        constraint_gradient_matrix,
        {
            {-1., 0., 0., 0.210962, 0., -0.617251},     // row 1
            {0., -1., 0., -0.00416521, 0., -0.775967},  // row 2
            {0., 0., -1., 0.977485, 0., 0.129909}       // row 3
        }
    );
}

TEST(HeavyTopProblemFromBrulsAndCardona2010PaperTest, CalculateIterationMatrix) {
    auto M = MassMatrix(15., Vector(0.234375, 0.46875, 0.234375));
    auto mass_matrix = M.GetMassMatrix();
    auto inertia_matrix = M.GetMomentOfInertiaMatrix();
    auto rotation_matrix = create_matrix(
        {{0.617251, -0.757955, 0.210962},
         {0.775967, 0.63076, -0.00416521},
         {-0.129909, 0.166271, 0.977485}}
    );
    auto angular_velocity_vector = create_vector({0.3, 0.1, 0.8});
    auto position_vector = create_vector({0., 1., 0.});
    auto lagrange_multipliers = create_vector({1., 2., 3.});
    const auto BETA_PRIME = 1.;
    const auto GAMMA_PRIME = 2.;

    auto iteration_matrix = heavy_top_iteration_matrix(
        BETA_PRIME, GAMMA_PRIME, mass_matrix, inertia_matrix, rotation_matrix,
        angular_velocity_vector, position_vector, lagrange_multipliers
    );

    expect_kokkos_view_2D_equal(
        iteration_matrix,
        {
            {15., 0., 0., 0., 0., 0., -1., 0., 0.},                                       // row 1
            {0., 15., 0., 0., 0., 0., 0., -1., 0.},                                       // row 2
            {0., 0., 15., 0., 0., 0., 0., 0., -1.},                                       // row 3
            {0., 0., 0., -0.768003, 1.404458, -0.046875, 0.210962, -0.004165, 0.977485},  // row 4
            {0., 0., 0., 0., 0.46875, 0., 0., 0., 0.},                                    // row 5
            {0., 0., 0., 0.046875, 3.275712, -0.768003, -0.617251, -0.775967, 0.129909},  // row 6
            {-1., 0., 0., 0.210962, 0., -0.617251, 0., 0., 0.},                           // row 7
            {0., -1., 0., -0.004165, 0., -0.775967, 0., 0., 0.},                          // row 8
            {0., 0., -1., 0.977485, 0., 0.129909, 0., 0., 0.}                             // row 9
        }
    );
}

}  // namespace openturbine::rigid_pendulum::tests
