#include <limits>

#include <gtest/gtest.h>

#include "src/gebt_poc/gen_alpha_2D.h"
#include "tests/unit_tests/gen_alpha_poc/test_utilities.h"

namespace openturbine::gebt_poc::tests {

TEST(TimeIntegratorTest, GetTimeIntegratorType) {
    auto time_integrator = GeneralizedAlphaTimeIntegrator(
        0.5, 0.5, 0.25, 0.5, gen_alpha_solver::TimeStepper(0., 1.0, 10)
    );

    EXPECT_EQ(time_integrator.GetType(), TimeIntegratorType::kGeneralizedAlpha);
}

TEST(TimeIntegratorTest, AdvanceAnalysisTimeByNumberOfSteps) {
    auto time_integrator = GeneralizedAlphaTimeIntegrator(
        0.5, 0.5, 0.25, 0.5, gen_alpha_solver::TimeStepper(0., 1.0, 10)
    );

    EXPECT_EQ(time_integrator.GetTimeStepper().GetCurrentTime(), 0.);

    auto q0 = gen_alpha_solver::create_matrix({{1., 1., 1., 1., 1., 1., 1.}});
    auto v0 = gen_alpha_solver::create_matrix({{2., 2., 2., 2., 2., 2.}});
    auto a0 = gen_alpha_solver::create_matrix({{3., 3., 3., 3., 3., 3.}});
    auto aa0 = gen_alpha_solver::create_matrix({{4., 4., 4., 4., 4., 4.}});
    auto initial_state = State{q0, v0, a0, aa0};

    size_t n_lagrange_mults{0};
    std::shared_ptr<LinearizationParameters> unity_linearization_parameters =
        std::make_shared<UnityLinearizationParameters>();

    auto state_history =
        time_integrator.Integrate(initial_state, n_lagrange_mults, unity_linearization_parameters);

    EXPECT_EQ(time_integrator.GetTimeStepper().GetCurrentTime(), 10.0);
    EXPECT_EQ(state_history.size(), 11);
    EXPECT_LE(
        time_integrator.GetTimeStepper().GetNumberOfIterations(),
        time_integrator.GetTimeStepper().GetMaximumNumberOfIterations()
    );
    EXPECT_LE(
        time_integrator.GetTimeStepper().GetTotalNumberOfIterations(),
        time_integrator.GetTimeStepper().GetNumberOfSteps() *
            time_integrator.GetTimeStepper().GetMaximumNumberOfIterations()
    );
}

TEST(TimeIntegratorTest, TestUpdateGeneralizedCoordinates) {
    auto time_integrator = GeneralizedAlphaTimeIntegrator(
        0.5, 0.5, 0.25, 0.5, gen_alpha_solver::TimeStepper(0., 1.0, 10)
    );

    auto gen_coords = gen_alpha_solver::create_matrix({{0., -1., 0., 1., 0., 0., 0.}});
    auto delta_gen_coords = gen_alpha_solver::create_matrix({{1., 1., 1., 1., 2., 3.}});
    auto gen_coords_next = gen_alpha_solver::create_matrix({{0., 0., 0., 0., 0., 0., 0.}});
    time_integrator.UpdateGeneralizedCoordinates(gen_coords, delta_gen_coords, gen_coords_next);

    gen_alpha_solver::Vector r1{0., -1., 0.};
    gen_alpha_solver::Vector r2{1., 1., 1.};
    gen_alpha_solver::Vector position = r1 + r2;

    gen_alpha_solver::Quaternion q1{1., 0., 0., 0.};
    gen_alpha_solver::Vector rotation_vector{1., 2., 3.};
    auto q2 = quaternion_from_rotation_vector(rotation_vector);
    gen_alpha_solver::Quaternion orientation = q1 * q2;

    gen_alpha_solver::tests::expect_kokkos_view_2D_equal(
        gen_coords_next,
        {{
            position.GetXComponent(),          // component 1
            position.GetYComponent(),          // component 2
            position.GetZComponent(),          // component 3
            orientation.GetScalarComponent(),  // component 4
            orientation.GetXComponent(),       // component 5
            orientation.GetYComponent(),       // component 6
            orientation.GetZComponent()        // component 7
        }}
    );
}

TEST(TimeIntegratorTest, ExpectConvergedSolution) {
    auto tol = GeneralizedAlphaTimeIntegrator::kConvergenceTolerance;
    auto residual = gen_alpha_solver::create_vector({tol * 1e-1, tol * 2e-1, tol * 3e-1});
    auto time_integrator = GeneralizedAlphaTimeIntegrator();
    auto converged = time_integrator.IsConverged(residual);

    EXPECT_TRUE(converged);
}

TEST(TimeIntegratorTest, ExpectNonConvergedSolution) {
    auto tol = GeneralizedAlphaTimeIntegrator::kConvergenceTolerance;
    auto residual = gen_alpha_solver::create_vector({tol * 1e1, tol * 2e1, tol * 3e1});
    auto time_integrator = GeneralizedAlphaTimeIntegrator();
    auto converged = time_integrator.IsConverged(residual);

    EXPECT_FALSE(converged);
}

TEST(GeneralizedAlphaTimeIntegratorTest, GetDefaultGAConstants) {
    auto time_integrator = GeneralizedAlphaTimeIntegrator();

    EXPECT_EQ(time_integrator.GetAlphaF(), 0.5);
    EXPECT_EQ(time_integrator.GetAlphaM(), 0.5);
    EXPECT_EQ(time_integrator.GetBeta(), 0.25);
    EXPECT_EQ(time_integrator.GetGamma(), 0.5);
}

TEST(GeneralizedAlphaTimeIntegratorTest, GetSuppliedGAConstants) {
    auto time_integrator = GeneralizedAlphaTimeIntegrator(0.11, 0.29, 0.47, 0.93);

    EXPECT_EQ(time_integrator.GetAlphaF(), 0.11);
    EXPECT_EQ(time_integrator.GetAlphaM(), 0.29);
    EXPECT_EQ(time_integrator.GetBeta(), 0.47);
    EXPECT_EQ(time_integrator.GetGamma(), 0.93);
}

TEST(TimeIntegratorTest, AlphaStepSolutionAfterOneAndTwoIncsWithZeroAcceleration) {
    auto time_integrator =
        GeneralizedAlphaTimeIntegrator(0., 0., 0.5, 1., gen_alpha_solver::TimeStepper(0., 1., 2, 1));

    auto q0 = gen_alpha_solver::create_matrix({{0., 0., 0., 0., 0., 0., 0.}});
    auto v0 = gen_alpha_solver::create_matrix({{0., 0., 0., 0., 0., 0.}});
    auto a0 = gen_alpha_solver::create_matrix({{0., 0., 0., 0., 0., 0.}});
    auto aa0 = gen_alpha_solver::create_matrix({{0., 0., 0., 0., 0., 0.}});
    auto initial_state = State{q0, v0, a0, aa0};

    size_t n_lagrange_mults{0};
    std::shared_ptr<LinearizationParameters> unity_linearization_parameters =
        std::make_shared<UnityLinearizationParameters>();

    auto results =
        time_integrator.Integrate(initial_state, n_lagrange_mults, unity_linearization_parameters);

    EXPECT_EQ(time_integrator.GetTimeStepper().GetNumberOfIterations(), 1);
    EXPECT_EQ(time_integrator.GetTimeStepper().GetTotalNumberOfIterations(), 2);
    EXPECT_EQ(results.size(), 3);

    // We expect the results State to contain the following values after one increments
    // via hand calculations
    auto first_state = results[1];
    gen_alpha_solver::tests::expect_kokkos_view_2D_equal(
        first_state.generalized_coordinates, {{0., 0., 0., 0., 0., 0., 0.}}
    );
    gen_alpha_solver::tests::expect_kokkos_view_2D_equal(
        first_state.velocity, {{-2., -2., -2., -2., -2., -2.}}
    );
    gen_alpha_solver::tests::expect_kokkos_view_2D_equal(
        first_state.acceleration, {{-2., -2., -2., -2., -2., -2.}}
    );
    gen_alpha_solver::tests::expect_kokkos_view_2D_equal(
        first_state.algorithmic_acceleration, {{-2., -2., -2., -2., -2., -2.}}
    );

    // We expect the results state to contain the following values after two increments
    // via hand calculations
    auto final_state = results.back();
    gen_alpha_solver::tests::expect_kokkos_view_2D_equal(
        final_state.generalized_coordinates, {{-2., -2., -2., 0., 0., 0., 0.}}
    );
    gen_alpha_solver::tests::expect_kokkos_view_2D_equal(
        final_state.velocity, {{-4., -4., -4., -4., -4., -4.}}
    );
    gen_alpha_solver::tests::expect_kokkos_view_2D_equal(
        final_state.acceleration, {{-2., -2., -2., -2., -2., -2.}}
    );
    gen_alpha_solver::tests::expect_kokkos_view_2D_equal(
        final_state.algorithmic_acceleration, {{-2., -2., -2., -2., -2., -2.}}
    );
}

TEST(TimeIntegratorTest, AlphaStepSolutionAfterOneIncWithNonZeroInitialState) {
    auto time_integrator =
        GeneralizedAlphaTimeIntegrator(0., 0., 0.5, 1., gen_alpha_solver::TimeStepper(0., 1., 1, 1));

    auto q0 = gen_alpha_solver::create_matrix({{0., 0., 0., 0., 0., 0., 0.}});
    auto v0 = gen_alpha_solver::create_matrix({{1., 2., 3., 4., 5., 6.}});
    ;
    auto a0 = gen_alpha_solver::create_matrix({{1., 2., 3., 4., 5., 6.}});
    ;
    auto aa0 = gen_alpha_solver::create_matrix({{1., 2., 3., 4., 5., 6.}});
    ;
    auto initial_state = State{q0, v0, a0, aa0};

    size_t n_lagrange_mults{0};
    std::shared_ptr<LinearizationParameters> unity_linearization_parameters =
        std::make_shared<UnityLinearizationParameters>();

    auto results =
        time_integrator.Integrate(initial_state, n_lagrange_mults, unity_linearization_parameters);

    EXPECT_EQ(time_integrator.GetTimeStepper().GetNumberOfIterations(), 1);
    EXPECT_EQ(time_integrator.GetTimeStepper().GetTotalNumberOfIterations(), 1);

    auto final_state = results.back();

    // We expect the final state to contain the following values after one increment
    // via hand calculations
    gen_alpha_solver::tests::expect_kokkos_view_2D_equal(
        final_state.generalized_coordinates, {{1., 2., 3., 0., 0., 0., 0.}}
    );
    gen_alpha_solver::tests::expect_kokkos_view_2D_equal(
        final_state.velocity, {{-1., 0., 1., 2., 3., 4.}}
    );
    gen_alpha_solver::tests::expect_kokkos_view_2D_equal(
        final_state.acceleration, {{-2., -2., -2., -2., -2., -2.}}
    );
    gen_alpha_solver::tests::expect_kokkos_view_2D_equal(
        final_state.algorithmic_acceleration, {{-2., -2., -2., -2., -2., -2.}}
    );
}

TEST(TimeIntegratorTest, AlphaStepSolutionOfHeavyTopAfterOneInc) {
    // Initial State for the heavy top problem
    auto q0 = gen_alpha_solver::create_matrix({{1., 1., 1., 1., 1., 1., 1.}});
    auto v0 = gen_alpha_solver::create_matrix({{2., 2., 2., 2., 2., 2.}});
    auto a0 = gen_alpha_solver::create_matrix({{3., 3., 3., 3., 3., 3.}});
    auto lag_mult = gen_alpha_solver::create_matrix({{4., 4., 4.}});
    auto aa0 = gen_alpha_solver::create_matrix({{5., 5., 5., 5., 5., 5.}});

    auto initial_state = State{q0, v0, a0, aa0};

    // Calculate properties for the time integrator
    double initial_time{0.};
    double final_time{0.1};
    double time_step{0.1};
    size_t num_steps = size_t(final_time / time_step);
    size_t max_iterations{1};

    auto time_stepper =
        gen_alpha_solver::TimeStepper(initial_time, time_step, num_steps, max_iterations);

    // Calculate the generalized alpha parameters
    auto rho_inf = 0.5;
    auto alpha_m = (2. * rho_inf - 1.) / (rho_inf + 1.);
    auto alpha_f = rho_inf / (rho_inf + 1.);
    auto gamma = 0.5 + alpha_f - alpha_m;
    auto beta = 0.25 * std::pow(gamma + 0.5, 2);

    auto time_integrator =
        GeneralizedAlphaTimeIntegrator(alpha_f, alpha_m, beta, gamma, time_stepper, true);

    // Calculate the required properties and initial conditions for the heavy top problem
    std::shared_ptr<LinearizationParameters> unity_linearization_parameters =
        std::make_shared<UnityLinearizationParameters>();

    // Perform the time integration
    auto results =
        time_integrator.Integrate(initial_state, lag_mult.size(), unity_linearization_parameters);

    auto final_state = results.back();

    // We expect the final state to contain the following values after one increment
    // via a pilot fortran code
    gen_alpha_solver::tests::expect_kokkos_view_2D_equal(
        final_state.generalized_coordinates,
        {{1.207222, 1.207222, 1.207222, 0.674773, 1.086996, 1.086996, 1.086996}}
    );
    gen_alpha_solver::tests::expect_kokkos_view_2D_equal(
        final_state.velocity,
        {{-16.583333, -16.583333, -16.583333, -16.583333, -16.583333, -16.583333}}
    );
    gen_alpha_solver::tests::expect_kokkos_view_2D_equal(
        final_state.acceleration, {{-337.5, -337.5, -337.5, -337.5, -337.5, -337.5}}
    );
    gen_alpha_solver::tests::expect_kokkos_view_2D_equal(
        final_state.algorithmic_acceleration, {{-224., -224., -224., -224., -224., -224.}}
    );
}

}  // namespace openturbine::gebt_poc::tests
