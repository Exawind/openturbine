#include <gtest/gtest.h>

#include "src/gebt_poc/gebt_generalized_alpha_time_integrator.h"
#include "tests/unit_tests/gen_alpha_poc/test_utilities.h"

TEST(GEBT_TimeIntegratorTest, AlphaStepSolutionAfterOneAndTwoIncsWithZeroAcceleration) {
    using namespace openturbine::gebt_poc;
    auto stepper = CreateBasicStepper();

    auto mesh = Create1DMesh(1, 1);
    auto field_data = FieldData(mesh, 1);
    constexpr auto lie_group_size = 7;
    constexpr auto lie_algebra_size = 6;

    Kokkos::parallel_for(
        mesh.GetNumberOfNodes(),
        KOKKOS_LAMBDA(int node) {
            auto coordinates = field_data.GetNodalData<Field::Coordinates>(node);
            for (int i = 0; i < lie_group_size; ++i) {
                coordinates(i) = 0.;
            }

            auto velocity = field_data.GetNodalData<Field::Velocity>(node);
            auto acceleration = field_data.GetNodalData<Field::Acceleration>(node);
            auto algo_acceleration = field_data.GetNodalData<Field::AlgorithmicAcceleration>(node);
            for (int i = 0; i < lie_algebra_size; ++i) {
                velocity(i) = 0.;
                acceleration(i) = 0.;
                algo_acceleration(i) = 0.;
            }
        }
    );

    std::size_t n_lagrange_mults = 0;

    auto time_step = 1.;
    auto max_nonlinear_iterations = 1;

    {
        [[maybe_unused]] bool step_converged =
            stepper.Step(mesh, field_data, n_lagrange_mults, time_step, max_nonlinear_iterations);

        using openturbine::gen_alpha_solver::tests::expect_kokkos_view_1D_equal;
        auto coordinates = Kokkos::View<double*>("coordinates", lie_group_size);
        Kokkos::deep_copy(coordinates, field_data.GetNodalData<Field::Coordinates>(0));
        expect_kokkos_view_1D_equal(coordinates, {0., 0., 0., 0., 0., 0., 0.});

        auto velocity = Kokkos::View<double*>("velocity", lie_algebra_size);
        Kokkos::deep_copy(velocity, field_data.GetNodalData<Field::Velocity>(0));
        expect_kokkos_view_1D_equal(velocity, {-2., -2., -2., -2., -2., -2.});

        auto acceleration = Kokkos::View<double*>("acceleration", lie_algebra_size);
        Kokkos::deep_copy(acceleration, field_data.GetNodalData<Field::Acceleration>(0));
        expect_kokkos_view_1D_equal(acceleration, {-2., -2., -2., -2., -2., -2.});

        auto algo_acceleration = Kokkos::View<double*>("algo acceleration", lie_algebra_size);
        Kokkos::deep_copy(
            algo_acceleration, field_data.GetNodalData<Field::AlgorithmicAcceleration>(0)
        );
        expect_kokkos_view_1D_equal(algo_acceleration, {-2., -2., -2., -2., -2., -2.});
    }
    {
        [[maybe_unused]] bool step_converged =
            stepper.Step(mesh, field_data, n_lagrange_mults, time_step, max_nonlinear_iterations);

        using openturbine::gen_alpha_solver::tests::expect_kokkos_view_1D_equal;
        auto coordinates = Kokkos::View<double*>("coordinates", lie_group_size);
        Kokkos::deep_copy(coordinates, field_data.GetNodalData<Field::Coordinates>(0));
        expect_kokkos_view_1D_equal(coordinates, {-2., -2., -2., 0., 0., 0., 0.});

        auto velocity = Kokkos::View<double*>("velocity", lie_algebra_size);
        Kokkos::deep_copy(velocity, field_data.GetNodalData<Field::Velocity>(0));
        expect_kokkos_view_1D_equal(velocity, {-4., -4., -4., -4., -4., -4.});

        auto acceleration = Kokkos::View<double*>("acceleration", lie_algebra_size);
        Kokkos::deep_copy(acceleration, field_data.GetNodalData<Field::Acceleration>(0));
        expect_kokkos_view_1D_equal(acceleration, {-2., -2., -2., -2., -2., -2.});

        auto algo_acceleration = Kokkos::View<double*>("algo acceleration", lie_algebra_size);
        Kokkos::deep_copy(
            algo_acceleration, field_data.GetNodalData<Field::AlgorithmicAcceleration>(0)
        );
        expect_kokkos_view_1D_equal(algo_acceleration, {-2., -2., -2., -2., -2., -2.});
    }
}

TEST(GEBT_TimeIntegratorTest, AlphaStepSolutionAfterOneIncWithNonZeroInitialState) {
    using namespace openturbine::gebt_poc;
    auto stepper = CreateBasicStepper();

    auto mesh = Create1DMesh(1, 1);
    auto field_data = FieldData(mesh, 1);
    constexpr auto lie_group_size = 7;
    constexpr auto lie_algebra_size = 6;

    Kokkos::parallel_for(
        mesh.GetNumberOfNodes(),
        KOKKOS_LAMBDA(int node) {
            auto coordinates = field_data.GetNodalData<Field::Coordinates>(node);
            for (int i = 0; i < lie_group_size; ++i) {
                coordinates(i) = 0.;
            }

            auto velocity = field_data.GetNodalData<Field::Velocity>(node);
            auto acceleration = field_data.GetNodalData<Field::Acceleration>(node);
            auto algo_acceleration = field_data.GetNodalData<Field::AlgorithmicAcceleration>(node);
            for (int i = 0; i < lie_algebra_size; ++i) {
                velocity(i) = i + 1.;
                acceleration(i) = i + 1.;
                algo_acceleration(i) = i + 1.;
            }
        }
    );

    std::size_t n_lagrange_mults = 0;

    auto time_step = 1.;
    auto max_nonlinear_iterations = 1;
    [[maybe_unused]] bool step_converged =
        stepper.Step(mesh, field_data, n_lagrange_mults, time_step, max_nonlinear_iterations);

    using openturbine::gen_alpha_solver::tests::expect_kokkos_view_1D_equal;
    auto coordinates = Kokkos::View<double*>("coordinates", lie_group_size);
    Kokkos::deep_copy(coordinates, field_data.GetNodalData<Field::Coordinates>(0));
    expect_kokkos_view_1D_equal(coordinates, {1., 2., 3., 0., 0., 0., 0.});

    auto velocity = Kokkos::View<double*>("velocity", lie_algebra_size);
    Kokkos::deep_copy(velocity, field_data.GetNodalData<Field::Velocity>(0));
    expect_kokkos_view_1D_equal(velocity, {-1., 0., 1., 2., 3., 4.});

    auto acceleration = Kokkos::View<double*>("acceleration", lie_algebra_size);
    Kokkos::deep_copy(acceleration, field_data.GetNodalData<Field::Acceleration>(0));
    expect_kokkos_view_1D_equal(acceleration, {-2., -2., -2., -2., -2., -2.});

    auto algo_acceleration = Kokkos::View<double*>("algo acceleration", lie_algebra_size);
    Kokkos::deep_copy(algo_acceleration, field_data.GetNodalData<Field::AlgorithmicAcceleration>(0));
    expect_kokkos_view_1D_equal(algo_acceleration, {-2., -2., -2., -2., -2., -2.});
}

TEST(GEBT_TimeIntegratorTest, AlphaStepSolutionOfHeavyTopAfterOneInc) {
    using namespace openturbine::gebt_poc;

    auto mesh = Create1DMesh(1, 1);
    auto field_data = FieldData(mesh, 1);
    constexpr auto lie_group_size = 7;
    constexpr auto lie_algebra_size = 6;

    Kokkos::parallel_for(
        mesh.GetNumberOfNodes(),
        KOKKOS_LAMBDA(int node) {
            auto coordinates = field_data.GetNodalData<Field::Coordinates>(node);
            for (int i = 0; i < lie_group_size; ++i) {
                coordinates(i) = 1.;
            }

            auto velocity = field_data.GetNodalData<Field::Velocity>(node);
            auto acceleration = field_data.GetNodalData<Field::Acceleration>(node);
            auto algo_acceleration = field_data.GetNodalData<Field::AlgorithmicAcceleration>(node);
            for (int i = 0; i < lie_algebra_size; ++i) {
                velocity(i) = 2.;
                acceleration(i) = 3.;
                algo_acceleration(i) = 5.;
            }
        }
    );

    // Calculate properties for the time integrator
    double time_step = 0.1;
    std::size_t max_nonlinear_iterations = 1;
    std::size_t n_lagrange_mults = 3;

    // Calculate the generalized alpha parameters
    auto rho_inf = 0.5;
    auto alpha_m = (2. * rho_inf - 1.) / (rho_inf + 1.);
    auto alpha_f = rho_inf / (rho_inf + 1.);
    auto gamma = 0.5 + alpha_f - alpha_m;
    auto beta = 0.25 * std::pow(gamma + 0.5, 2);

    auto stepper = CreateStepper(alpha_f, alpha_m, beta, gamma, true);

    [[maybe_unused]] bool step_converged =
        stepper.Step(mesh, field_data, n_lagrange_mults, time_step, max_nonlinear_iterations);

    using openturbine::gen_alpha_solver::tests::expect_kokkos_view_1D_equal;
    auto coordinates = Kokkos::View<double*>("coordinates", lie_group_size);
    Kokkos::deep_copy(coordinates, field_data.GetNodalData<Field::Coordinates>(0));
    expect_kokkos_view_1D_equal(
        coordinates, {1.207222, 1.207222, 1.207222, 0.674773, 1.086996, 1.086996, 1.086996}
    );

    auto velocity = Kokkos::View<double*>("velocity", lie_algebra_size);
    Kokkos::deep_copy(velocity, field_data.GetNodalData<Field::Velocity>(0));
    expect_kokkos_view_1D_equal(
        velocity, {-16.583333, -16.583333, -16.583333, -16.583333, -16.583333, -16.583333}
    );

    auto acceleration = Kokkos::View<double*>("acceleration", lie_algebra_size);
    Kokkos::deep_copy(acceleration, field_data.GetNodalData<Field::Acceleration>(0));
    expect_kokkos_view_1D_equal(acceleration, {-337.5, -337.5, -337.5, -337.5, -337.5, -337.5});

    auto algo_acceleration = Kokkos::View<double*>("algo acceleration", lie_algebra_size);
    Kokkos::deep_copy(algo_acceleration, field_data.GetNodalData<Field::AlgorithmicAcceleration>(0));
    expect_kokkos_view_1D_equal(algo_acceleration, {-224., -224., -224., -224., -224., -224.});
}
