#include <iostream>

#include <gtest/gtest.h>

#include "src/gebt_poc/dynamic_beam_element.h"
#include "src/gebt_poc/gen_alpha_2D.h"
#include "src/gebt_poc/solver.h"
#include "tests/unit_tests/gen_alpha_poc/test_utilities.h"

namespace openturbine::gebt_poc::tests {

struct PopulatePositionVectors {
    Kokkos::View<double[35]> position_vectors;
    KOKKOS_FUNCTION
    void operator()(std::size_t) const {
        // node 1
        position_vectors(0) = 1.;
        position_vectors(1) = 0.;
        position_vectors(2) = 0.;
        position_vectors(3) = 1.;
        position_vectors(4) = 0.;
        position_vectors(5) = 0.;
        position_vectors(6) = 0.;
        // node 2
        position_vectors(7) = 2.7267316464601146;
        position_vectors(8) = 0.;
        position_vectors(9) = 0.;
        position_vectors(10) = 1.;
        position_vectors(11) = 0.;
        position_vectors(12) = 0.;
        position_vectors(13) = 0.;
        // node 3
        position_vectors(14) = 6.;
        position_vectors(15) = 0.;
        position_vectors(16) = 0.;
        position_vectors(17) = 1.;
        position_vectors(18) = 0.;
        position_vectors(19) = 0.;
        position_vectors(20) = 0.;
        // node 4
        position_vectors(21) = 9.273268353539885;
        position_vectors(22) = 0.;
        position_vectors(23) = 0.;
        position_vectors(24) = 1.;
        position_vectors(25) = 0.;
        position_vectors(26) = 0.;
        position_vectors(27) = 0.;
        // node 5
        position_vectors(28) = 11.;
        position_vectors(29) = 0.;
        position_vectors(30) = 0.;
        position_vectors(31) = 1.;
        position_vectors(32) = 0.;
        position_vectors(33) = 0.;
        position_vectors(34) = 0.;
    }
};

TEST(DynamicBeamTest, DynamicAnalysisWithZeroForce) {
    auto position_vectors = Kokkos::View<double[35]>("position_vectors");
    Kokkos::parallel_for(1, PopulatePositionVectors{position_vectors});

    // Stiffness matrix for uniform composite beam section (in material csys)
    auto stiffness = gen_alpha_solver::create_matrix({
        {1.36817e6, 0., 0., 0., 0., 0.},      // row 1
        {0., 88560., 0., 0., 0., 0.},         // row 2
        {0., 0., 38780., 0., 0., 0.},         // row 3
        {0., 0., 0., 16960., 17610., -351.},  // row 4
        {0., 0., 0., 17610., 59120., -370.},  // row 5
        {0., 0., 0., -351., -370., 141470.}   // row 6
    });
    auto stiffness_matrix = StiffnessMatrix(stiffness);

    auto mm = gen_alpha_solver::create_matrix({
        {2., 0., 0., 0., 0.6, -0.4},  // row 1
        {0., 2., 0., -0.6, 0., 0.2},  // row 2
        {0., 0., 2., 0.4, -0.2, 0.},  // row 3
        {0., -0.6, 0.4, 1., 2., 3.},  // row 4
        {0.6, 0., -0.2, 2., 4., 6.},  // row 5
        {-0.4, 0.2, 0., 3., 6., 9.}   // row 6
    });
    auto mass_matrix = MassMatrix(mm);

    auto quadrature = UserDefinedQuadrature(
        {-0.9491079123427585, -0.7415311855993945, -0.4058451513773972, 0., 0.4058451513773972,
         0.7415311855993945, 0.9491079123427585},
        {0.1294849661688697, 0.2797053914892766, 0.3818300505051189, 0.4179591836734694,
         0.3818300505051189, 0.2797053914892766, 0.1294849661688697}
    );

    auto gen_coords = gen_alpha_solver::create_matrix({
        {0., 0., 0., 1., 0., 0., 0.},  // node 1
        {0., 0., 0., 1., 0., 0., 0.},  // node 2
        {0., 0., 0., 1., 0., 0., 0.},  // node 3
        {0., 0., 0., 1., 0., 0., 0.},  // node 4
        {0., 0., 0., 1., 0., 0., 0.}   // node 5
    });

    auto v = gen_alpha_solver::create_matrix(
        {{0., 0., 0., 0., 0., 0.},  // node 1
         {0., 0., 0., 0., 0., 0.},  // node 2
         {0., 0., 0., 0., 0., 0.},  // node 3
         {0., 0., 0., 0., 0., 0.},  // node 4
         {0., 0., 0., 0., 0., 0.}}  // node 5
    );

    auto velocity = v;
    auto acceleration = v;
    auto algo_acceleration = v;
    auto initial_state = State(gen_coords, velocity, acceleration, algo_acceleration);

    auto lagrange_mults = gen_alpha_solver::create_vector({0., 0., 0., 0., 0., 0.});
    auto time_integrator = GeneralizedAlphaTimeIntegrator(
        0., 0., 0.5, 1., gen_alpha_solver::TimeStepper(0., 1., 1, 20), false
    );
    std::shared_ptr<LinearizationParameters> dynamic_beam_lin_params =
        std::make_shared<DynamicBeamLinearizationParameters>(
            position_vectors, stiffness_matrix, mass_matrix, quadrature
        );
    auto results =
        time_integrator.Integrate(initial_state, lagrange_mults.extent(0), dynamic_beam_lin_params);
    auto final_state = results.back();

    openturbine::gen_alpha_solver::tests::expect_kokkos_view_2D_equal(
        final_state.GetGeneralizedCoordinates(),
        {
            {0., 0., 0., 1., 0., 0., 0.},  // node 1
            {0., 0., 0., 1., 0., 0., 0.},  // node 2
            {0., 0., 0., 1., 0., 0., 0.},  // node 3
            {0., 0., 0., 1., 0., 0., 0.},  // node 4
            {0., 0., 0., 1., 0., 0., 0.}   // node 5
        }
    );
}

struct NonZeroValues_populate_position {
    Kokkos::View<double[35]> position_vectors;

    KOKKOS_FUNCTION
    void operator()(std::size_t) const {
        // node 1
        position_vectors(0) = 0.;
        position_vectors(1) = 0.;
        position_vectors(2) = 0.;
        position_vectors(3) = 1.;
        position_vectors(4) = 0.;
        position_vectors(5) = 0.;
        position_vectors(6) = 0.;
        // node 2
        position_vectors(7) = 1.7267316464601146;
        position_vectors(8) = 0.;
        position_vectors(9) = 0.;
        position_vectors(10) = 1.;
        position_vectors(11) = 0.;
        position_vectors(12) = 0.;
        position_vectors(13) = 0.;
        // node 3
        position_vectors(14) = 5.;
        position_vectors(15) = 0.;
        position_vectors(16) = 0.;
        position_vectors(17) = 1.;
        position_vectors(18) = 0.;
        position_vectors(19) = 0.;
        position_vectors(20) = 0.;
        // node 4
        position_vectors(21) = 8.273268353539885;
        position_vectors(22) = 0.;
        position_vectors(23) = 0.;
        position_vectors(24) = 1.;
        position_vectors(25) = 0.;
        position_vectors(26) = 0.;
        position_vectors(27) = 0.;
        // node 5
        position_vectors(28) = 10.;
        position_vectors(29) = 0.;
        position_vectors(30) = 0.;
        position_vectors(31) = 1.;
        position_vectors(32) = 0.;
        position_vectors(33) = 0.;
        position_vectors(34) = 0.;
    }
};

TEST(DynamicBeamTest, DynamicAnalysisCatileverWithConstantForceAtTip) {
    // Set up the initial state for the problem
    auto gen_coords = gen_alpha_solver::create_matrix({
        {0., 0., 0., 1., 0., 0., 0.},  // node 1
        {0., 0., 0., 1., 0., 0., 0.},  // node 2
        {0., 0., 0., 1., 0., 0., 0.},  // node 3
        {0., 0., 0., 1., 0., 0., 0.},  // node 4
        {0., 0., 0., 1., 0., 0., 0.}   // node 5
    });
    auto v = gen_alpha_solver::create_matrix(
        {{0., 0., 0., 0., 0., 0.},  // node 1
         {0., 0., 0., 0., 0., 0.},  // node 2
         {0., 0., 0., 0., 0., 0.},  // node 3
         {0., 0., 0., 0., 0., 0.},  // node 4
         {0., 0., 0., 0., 0., 0.}}  // node 5
    );
    auto velocity = v;
    auto acceleration = v;
    auto algo_acceleration = v;

    auto initial_state = State(gen_coords, velocity, acceleration, algo_acceleration);

    // Set up the linearization parameters for the problem
    auto stiffness = gen_alpha_solver::create_matrix({
        {1368.17e3, 0., 0., 0., 0., 0.},      // row 1
        {0., 88560., 0., 0., 0., 0.},         // row 2
        {0., 0., 38780., 0., 0., 0.},         // row 3
        {0., 0., 0., 16960., 17610., -351.},  // row 4
        {0., 0., 0., 17610., 59120., -370.},  // row 5
        {0., 0., 0., -351., -370., 141470.}   // row 6
    });
    auto stiffness_matrix = StiffnessMatrix(stiffness);

    auto mm = gen_alpha_solver::create_matrix({
        {8.538e-2, 0., 0., 0., 0., 0.},    // row 1
        {0., 8.538e-2, 0., 0., 0., 0.},    // row 2
        {0., 0., 8.538e-2, 0., 0., 0.},    // row 3
        {0., 0., 0., 1.4433e-2, 0., 0.},   // row 4
        {0., 0., 0., 0., 0.40972e-2, 0.},  // row 5
        {0., 0., 0., 0., 0., 1.0336e-2}    // row 6
    });
    auto mass_matrix = MassMatrix(mm);

    auto quadrature = UserDefinedQuadrature(
        {-0.9491079123427585, -0.7415311855993945, -0.4058451513773972, 0., 0.4058451513773972,
         0.7415311855993945, 0.9491079123427585},
        {0.1294849661688697, 0.2797053914892766, 0.3818300505051189, 0.4179591836734694,
         0.3818300505051189, 0.2797053914892766, 0.1294849661688697}
    );

    auto position_vectors =
        gen_alpha_solver::create_vector({// node 1
                                         0., 0., 0., 1., 0., 0., 0.,
                                         // node 2
                                         1.7267316464601146, 0., 0., 1., 0., 0., 0.,
                                         // node 3
                                         5., 0., 0., 1., 0., 0., 0.,
                                         // node 4
                                         8.273268353539885, 0., 0., 1., 0., 0., 0.,
                                         // node 5
                                         10., 0., 0., 1., 0., 0., 0.});

    auto external_forces = std::vector<Forces*>{};
    auto constant_tip_force =
        GeneralizedForces(gen_alpha_solver::create_vector({0., 0., 1., 0., 0., 0.}), 5);
    external_forces.push_back(&constant_tip_force);

    std::shared_ptr<LinearizationParameters> dynamic_beam_lin_params =
        std::make_shared<DynamicBeamLinearizationParameters>(
            position_vectors, stiffness_matrix, mass_matrix, quadrature, external_forces
        );

    auto time_stepper = gen_alpha_solver::TimeStepper(0., 0.005, 10, 10);

    // Calculate the generalized alpha parameters for rho_inf = 0
    auto rho_inf = 0.;
    auto alpha_m = (2. * rho_inf - 1.) / (rho_inf + 1.);
    auto alpha_f = rho_inf / (rho_inf + 1.);
    auto gamma = 0.5 + alpha_f - alpha_m;
    auto beta = 0.25 * std::pow(gamma + 0.5, 2);

    auto time_integrator = GeneralizedAlphaTimeIntegrator(
        alpha_f, alpha_m, beta, gamma, time_stepper, true, ProblemType::kDynamic
    );

    auto lagrange_mults = gen_alpha_solver::create_vector({0., 0., 0., 0., 0., 0.});
    auto results =
        time_integrator.Integrate(initial_state, lagrange_mults.extent(0), dynamic_beam_lin_params);
    auto final_state = results.back();
}

TEST(DynamicBeamTest, DynamicAnalysisCatileverWithSinusoidalForceAtTip) {
    // Set up the initial state for the problem
    auto gen_coords = gen_alpha_solver::create_matrix({
        {0., 0., 0., 1., 0., 0., 0.},  // node 1
        {0., 0., 0., 1., 0., 0., 0.},  // node 2
        {0., 0., 0., 1., 0., 0., 0.},  // node 3
        {0., 0., 0., 1., 0., 0., 0.},  // node 4
        {0., 0., 0., 1., 0., 0., 0.}   // node 5
    });
    auto v = gen_alpha_solver::create_matrix(
        {{0., 0., 0., 0., 0., 0.},  // node 1
         {0., 0., 0., 0., 0., 0.},  // node 2
         {0., 0., 0., 0., 0., 0.},  // node 3
         {0., 0., 0., 0., 0., 0.},  // node 4
         {0., 0., 0., 0., 0., 0.}}  // node 5
    );
    auto velocity = v;
    auto acceleration = v;
    auto algo_acceleration = v;

    auto initial_state = State(gen_coords, velocity, acceleration, algo_acceleration);

    // Set up the linearization parameters for the problem
    auto stiffness = gen_alpha_solver::create_matrix({
        {1368.17e3, 0., 0., 0., 0., 0.},      // row 1
        {0., 88560., 0., 0., 0., 0.},         // row 2
        {0., 0., 38780., 0., 0., 0.},         // row 3
        {0., 0., 0., 16960., 17610., -351.},  // row 4
        {0., 0., 0., 17610., 59120., -370.},  // row 5
        {0., 0., 0., -351., -370., 141470.}   // row 6
    });
    auto stiffness_matrix = StiffnessMatrix(stiffness);

    auto mm = gen_alpha_solver::create_matrix({
        {8.538e-2, 0., 0., 0., 0., 0.},    // row 1
        {0., 8.538e-2, 0., 0., 0., 0.},    // row 2
        {0., 0., 8.538e-2, 0., 0., 0.},    // row 3
        {0., 0., 0., 1.4433e-2, 0., 0.},   // row 4
        {0., 0., 0., 0., 0.40972e-2, 0.},  // row 5
        {0., 0., 0., 0., 0., 1.0336e-2}    // row 6
    });
    auto mass_matrix = MassMatrix(mm);

    auto quadrature = UserDefinedQuadrature(
        {-0.9491079123427585, -0.7415311855993945, -0.4058451513773972, 0., 0.4058451513773972,
         0.7415311855993945, 0.9491079123427585},
        {0.1294849661688697, 0.2797053914892766, 0.3818300505051189, 0.4179591836734694,
         0.3818300505051189, 0.2797053914892766, 0.1294849661688697}
    );

    auto position_vectors =
        gen_alpha_solver::create_vector({// node 1
                                         0., 0., 0., 1., 0., 0., 0.,
                                         // node 2
                                         1.7267316464601146, 0., 0., 1., 0., 0., 0.,
                                         // node 3
                                         5., 0., 0., 1., 0., 0., 0.,
                                         // node 4
                                         8.273268353539885, 0., 0., 1., 0., 0., 0.,
                                         // node 5
                                         10., 0., 0., 1., 0., 0., 0.});

    auto time_stepper = gen_alpha_solver::TimeStepper(0., 0.005, 10, 10);

    auto external_forces = std::vector<Forces*>{};
    auto create_sin_varying_force = [](double t) {
        return gen_alpha_solver::create_vector({0., 0., 100. * std::sin(10. * t), 0., 0., 0.});
    };
    auto time_varying_force = TimeVaryingForces(create_sin_varying_force, 5);
    external_forces.push_back(&time_varying_force);

    std::shared_ptr<LinearizationParameters> dynamic_beam_lin_params =
        std::make_shared<DynamicBeamLinearizationParameters>(
            position_vectors, stiffness_matrix, mass_matrix, quadrature, external_forces
        );

    // Calculate the generalized alpha parameters for rho_inf = 0
    auto rho_inf = 0.;
    auto alpha_m = (2. * rho_inf - 1.) / (rho_inf + 1.);
    auto alpha_f = rho_inf / (rho_inf + 1.);
    auto gamma = 0.5 + alpha_f - alpha_m;
    auto beta = 0.25 * std::pow(gamma + 0.5, 2);

    auto time_integrator = GeneralizedAlphaTimeIntegrator(
        alpha_f, alpha_m, beta, gamma, time_stepper, true, ProblemType::kDynamic
    );

    auto lagrange_mults = gen_alpha_solver::create_vector({0., 0., 0., 0., 0., 0.});
    auto results =
        time_integrator.Integrate(initial_state, lagrange_mults.extent(0), dynamic_beam_lin_params);
    auto final_state = results.back();
}

}  // namespace openturbine::gebt_poc::tests
