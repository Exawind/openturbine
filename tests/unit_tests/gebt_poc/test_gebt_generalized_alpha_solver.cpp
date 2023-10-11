#include <gtest/gtest.h>

#include "src/gebt_poc/gebt_generalized_alpha_time_integrator.h"
#include "tests/unit_tests/gen_alpha_poc/test_utilities.h"

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
    bool step_converged =
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
