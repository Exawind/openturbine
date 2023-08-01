#include <gtest/gtest.h>

#include "src/rigid_pendulum_poc/generalized_alpha_time_integrator.h"
#include "src/rigid_pendulum_poc/heavy_top.h"
#include "tests/unit_tests/rigid_pendulum_poc/test_utilities.h"

namespace openturbine::rigid_pendulum::tests {

TEST(HeavyTopProblemFromBrulsAndCardona2010PaperTest, CalculateGenCoordsResidualVector) {
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

    auto residual_vector = heavy_top_gen_coords_residual_vector(
        mass_matrix, rotation_matrix, acceleration_vector, gen_forces_vector, lagrange_multipliers,
        position_vector
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

TEST(HeavyTopProblemFromBrulsAndCardona2010PaperTest, CalculateConstraintsResidualVector) {
    auto rotation_matrix = create_matrix(
        {{0.617251, -0.757955, 0.210962},
         {0.775967, 0.63076, -0.00416521},
         {-0.129909, 0.166271, 0.977485}}
    );
    auto position_vector = create_vector({1., 2., 3.});
    auto ref_position_vector = create_vector({0., 1., 0.});

    auto residual_vector =
        heavy_top_constraints_residual_vector(rotation_matrix, position_vector, ref_position_vector);

    expect_kokkos_view_1D_equal(
        residual_vector,
        {
            -1.757955,  // row 1
            -1.369240,  // row 2
            -2.833729   // row 3
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
        heavy_top_tangent_stiffness_matrix(rotation_matrix, lagrange_multipliers, position_vector);

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
        heavy_top_constraint_gradient_matrix(rotation_matrix, position_vector);

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
        angular_velocity_vector, lagrange_multipliers, position_vector
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

TEST(HeavyTopProblemFromBrulsAndCardona2010PaperTest, CalculateTangentOperatorWithPhiAsZero) {
    auto psi = create_vector({0., 0., 0.});
    auto tangent_operator = heavy_top_tangent_operator(psi);

    expect_kokkos_view_2D_equal(
        tangent_operator,
        {
            {1., 0., 0., 0., 0., 0.},  // row 1
            {0., 1., 0., 0., 0., 0.},  // row 2
            {0., 0., 1., 0., 0., 0.},  // row 3
            {0., 0., 0., 1., 0., 0.},  // row 4
            {0., 0., 0., 0., 1., 0.},  // row 5
            {0., 0., 0., 0., 0., 1.}   // row 6
        }
    );
}

TEST(HeavyTopProblemFromBrulsAndCardona2010PaperTest, CalculateTangentOperatorWithPhiNotZero) {
    auto psi = create_vector({1., 2., 3.});
    auto tangent_operator = heavy_top_tangent_operator(psi);

    expect_kokkos_view_2D_equal(
        tangent_operator,
        {
            {1., 0., 0., 0., 0., 0.},                                                      // row 1
            {0., 1., 0., 0., 0., 0.},                                                      // row 2
            {0., 0., 1., 0., 0., 0.},                                                      // row 3
            {0., 0., 0., -0.06871266098996709, 0.555552845761836, -0.014131010177901665},  // row 4
            {0., 0., 0., -0.2267181808418461, 0.1779133377000255, 0.6236305018139318},     // row 5
            {0., 0., 0., 0.5073830075578865, 0.36287349294603777, 0.5889566688500127}      // row 6
        }
    );
}

TEST(HeavyTopProblemFromBrulsAndCardona2010PaperTest, AlphaStepSolutionAfterOneInc) {
    // Calculate the required properties and initial conditions for the heavy top problem
    auto mass = 15.;
    auto mass_matrix = MassMatrix(mass, Vector(0.234375, 0.46875, 0.234375));

    auto gravity = Vector(0., 0., -9.81);
    auto forces = gravity * mass;

    auto angular_velocity = create_vector({0.3, 0.1, 0.8});
    auto J = mass_matrix.GetMomentOfInertiaMatrix();
    auto J_omega = multiply_matrix_with_vector(J, angular_velocity);
    auto angular_velocity_vector =
        Vector(angular_velocity(0), angular_velocity(1), angular_velocity(2));
    auto J_omega_vector = Vector(J_omega(0), J_omega(1), J_omega(2));
    auto moments = angular_velocity_vector.CrossProduct(J_omega_vector);

    auto gen_forces = GeneralizedForces(forces, moments);

    auto X0 = create_vector({0., 1., 0.});
    auto rot0 = create_matrix({{1., 0., 0.}, {0., 1., 0.}, {0., 0., 1.}});

    // Convert the above into initial State for the heavy top problem
    auto initial_position = multiply_matrix_with_vector(rot0, X0);
    auto initial_orientation =
        rotation_matrix_to_quaternion(RotationMatrix{{1., 0., 0.}, {0., 1., 0.}, {0., 0., 1.}});
    auto q0 = create_vector({
        initial_position(0),                       // component 1
        initial_position(1),                       // component 2
        initial_position(2),                       // component 3
        initial_orientation.GetScalarComponent(),  // component 4
        initial_orientation.GetXComponent(),       // component 5
        initial_orientation.GetYComponent(),       // component 6
        initial_orientation.GetZComponent()        // component 7
    });

    auto omega0 = Vector({0., 150., -4.61538});
    auto x_0 = Vector({0., 1., 0.});
    auto initial_velocity = omega0.CrossProduct(x_0);

    auto v0 = create_vector({
        initial_velocity.GetXComponent(),  // component 1
        initial_velocity.GetYComponent(),  // component 2
        initial_velocity.GetZComponent(),  // component 3
        omega0.GetXComponent(),            // component 4
        omega0.GetYComponent(),            // component 5
        omega0.GetZComponent()             // component 6
    });

    auto a0 =
        create_vector({0., -21.301732544400004, -30.960830769230938, 661.3461692307692, 0., 0.});
    auto aa0 = create_vector({0., 0., 0., 0., 0., 0.});  // algorithmic acceleration

    auto initial_state = State(q0, v0, a0, aa0);

    // Calculate properties for the time integrator
    double initial_time{0.};
    double final_time{0.02};
    double time_step{0.002};
    size_t num_steps = size_t(final_time / time_step);  // 10 steps
    size_t max_iterations{10};

    auto time_stepper = TimeStepper(initial_time, time_step, num_steps, max_iterations);

    // Calculate the generalized alpha parameters
    auto rho_inf = 0.6;
    auto alpha_m = (2. * rho_inf - 1.) / (rho_inf + 1.);
    auto alpha_f = rho_inf / (rho_inf + 1.);
    auto gamma = 0.5 + alpha_f - alpha_m;
    auto beta = 0.25 * std::pow(gamma + 0.5, 2);

    auto time_integrator = GeneralizedAlphaTimeIntegrator(
        alpha_f, alpha_m, beta, gamma, time_stepper, ProblemType::kHeavyTop, true
    );

    // Initialize the lagrange multipliers to zero
    auto lagrange_mults = create_vector({0., 0., 0.});

    // Perform the time integration
    auto results = time_integrator.Integrate(initial_state, mass_matrix, gen_forces, lagrange_mults);

    // Expected values of alpham, alphaf, beta, gamma, betap, gammap from prototype fortran code
    // 0.12499999999999997       0.37499999999999994       0.39062500000000000 0.75000000000000000
    // 896000.00000000000        960.00000000000000
    EXPECT_EQ(time_integrator.GetAlphaF(), 0.37499999999999994);
    EXPECT_EQ(time_integrator.GetAlphaM(), 0.12499999999999997);
    EXPECT_EQ(time_integrator.GetBeta(), 0.39062500000000000);
    EXPECT_EQ(time_integrator.GetGamma(), 0.75000000000000000);

    auto final_state = results.back();

    // We expect the final state to contain the following values after one increment
    // via a pilot fortran code
    expect_kokkos_view_1D_equal(
        final_state.GetGeneralizedCoordinates(),
        {0.091943, 0.995745, -0.006167, 0.070604, 0.045687, 0.996438, -0.006332}
    );
    expect_kokkos_view_1D_equal(
        final_state.GetVelocity(), {4.564702, -0.425498, -0.620310, 1.287734, 150., 4.522188}
    );
    expect_kokkos_view_1D_equal(
        final_state.GetAcceleration(),
        {-6.272347, -24.111949, -29.783310, -560.293932, 0., 244.069741}
    );
    expect_kokkos_view_1D_equal(
        final_state.GetAlgorithmicAcceleration(),
        {-4.850215, -21.808817, -30.587676, -616.404070, 0., 241.543883}
    );
}

}  // namespace openturbine::rigid_pendulum::tests
