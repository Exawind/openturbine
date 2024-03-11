#include <iostream>

#include <gtest/gtest.h>

#include "src/gebt_poc/dynamic_beam_element.h"
#include "src/gebt_poc/gen_alpha_2D.h"
#include "tests/unit_tests/gen_alpha_poc/test_utilities.h"

namespace openturbine::gebt_poc::tests {

struct PopulatePositionVectors {
    Kokkos::View<double[5][7]> position_vectors;
    KOKKOS_FUNCTION
    void operator()(std::size_t) const {
        // node 1
        position_vectors(0, 0) = 1.;
        position_vectors(0, 1) = 0.;
        position_vectors(0, 2) = 0.;
        position_vectors(0, 3) = 1.;
        position_vectors(0, 4) = 0.;
        position_vectors(0, 5) = 0.;
        position_vectors(0, 6) = 0.;
        // node 2
        position_vectors(1, 0) = 2.7267316464601146;
        position_vectors(1, 1) = 0.;
        position_vectors(1, 2) = 0.;
        position_vectors(1, 3) = 1.;
        position_vectors(1, 4) = 0.;
        position_vectors(1, 5) = 0.;
        position_vectors(1, 6) = 0.;
        // node 3
        position_vectors(2, 0) = 6.;
        position_vectors(2, 1) = 0.;
        position_vectors(2, 2) = 0.;
        position_vectors(2, 3) = 1.;
        position_vectors(2, 4) = 0.;
        position_vectors(2, 5) = 0.;
        position_vectors(2, 6) = 0.;
        // node 4
        position_vectors(3, 0) = 9.273268353539885;
        position_vectors(3, 1) = 0.;
        position_vectors(3, 2) = 0.;
        position_vectors(3, 3) = 1.;
        position_vectors(3, 4) = 0.;
        position_vectors(3, 5) = 0.;
        position_vectors(3, 6) = 0.;
        // node 5
        position_vectors(4, 0) = 11.;
        position_vectors(4, 1) = 0.;
        position_vectors(4, 2) = 0.;
        position_vectors(4, 3) = 1.;
        position_vectors(4, 4) = 0.;
        position_vectors(4, 5) = 0.;
        position_vectors(4, 6) = 0.;
    }
};

TEST(DynamicBeamTest, DynamicAnalysisWithZeroForce) {
    auto position_vectors = Kokkos::View<double[5][7]>("position_vectors");
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

    auto mass_matrix = gen_alpha_solver::create_matrix({
        {2., 0., 0., 0., 0.6, -0.4},  // row 1
        {0., 2., 0., -0.6, 0., 0.2},  // row 2
        {0., 0., 2., 0.4, -0.2, 0.},  // row 3
        {0., -0.6, 0.4, 1., 2., 3.},  // row 4
        {0.6, 0., -0.2, 2., 4., 6.},  // row 5
        {-0.4, 0.2, 0., 3., 6., 9.}   // row 6
    });

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

    auto velocity = gen_alpha_solver::create_matrix(
        {{0., 0., 0., 0., 0., 0.},  // node 1
         {0., 0., 0., 0., 0., 0.},  // node 2
         {0., 0., 0., 0., 0., 0.},  // node 3
         {0., 0., 0., 0., 0., 0.},  // node 4
         {0., 0., 0., 0., 0., 0.}}  // node 5
    );

    auto acceleration = gen_alpha_solver::create_matrix(
        {{0., 0., 0., 0., 0., 0.},  // node 1
         {0., 0., 0., 0., 0., 0.},  // node 2
         {0., 0., 0., 0., 0., 0.},  // node 3
         {0., 0., 0., 0., 0., 0.},  // node 4
         {0., 0., 0., 0., 0., 0.}}  // node 5
    );

    auto algo_acceleration = gen_alpha_solver::create_matrix(
        {{0., 0., 0., 0., 0., 0.},  // node 1
         {0., 0., 0., 0., 0., 0.},  // node 2
         {0., 0., 0., 0., 0., 0.},  // node 3
         {0., 0., 0., 0., 0., 0.},  // node 4
         {0., 0., 0., 0., 0., 0.}}  // node 5
    );

    auto initial_state = State{gen_coords, velocity, acceleration, algo_acceleration};

    auto lagrange_mults = gen_alpha_solver::create_vector({0., 0., 0., 0., 0., 0.});
    auto time_integrator = GeneralizedAlphaTimeIntegrator(
        0., 0., 0.5, 1., gen_alpha_solver::TimeStepper(0., 1., 1, 20), false
    );
    std::shared_ptr<LinearizationParameters> dynamic_beam_lin_params =
        std::make_shared<DynamicBeamLinearizationParameters>(
            position_vectors, stiffness, mass_matrix, quadrature
        );
    auto results =
        time_integrator.Integrate(initial_state, lagrange_mults.extent(0), dynamic_beam_lin_params);
    auto final_state = results.back();

    openturbine::gen_alpha_solver::tests::expect_kokkos_view_2D_equal(
        final_state.generalized_coordinates,
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
    auto velocity = gen_alpha_solver::create_matrix(
        {{0., 0., 0., 0., 0., 0.},  // node 1
         {0., 0., 0., 0., 0., 0.},  // node 2
         {0., 0., 0., 0., 0., 0.},  // node 3
         {0., 0., 0., 0., 0., 0.},  // node 4
         {0., 0., 0., 0., 0., 0.}}  // node 5
    );
    auto acceleration = gen_alpha_solver::create_matrix(
        {{0., 0., 0., 0., 0., 0.},  // node 1
         {0., 0., 0., 0., 0., 0.},  // node 2
         {0., 0., 0., 0., 0., 0.},  // node 3
         {0., 0., 0., 0., 0., 0.},  // node 4
         {0., 0., 0., 0., 0., 0.}}  // node 5
    );
    auto algo_acceleration = gen_alpha_solver::create_matrix(
        {{0., 0., 0., 0., 0., 0.},  // node 1
         {0., 0., 0., 0., 0., 0.},  // node 2
         {0., 0., 0., 0., 0., 0.},  // node 3
         {0., 0., 0., 0., 0., 0.},  // node 4
         {0., 0., 0., 0., 0., 0.}}  // node 5
    );

    auto initial_state = State{gen_coords, velocity, acceleration, algo_acceleration};

    // Set up the linearization parameters for the problem
    auto stiffness = gen_alpha_solver::create_matrix({
        {1368.17e3, 0., 0., 0., 0., 0.},      // row 1
        {0., 88560., 0., 0., 0., 0.},         // row 2
        {0., 0., 38780., 0., 0., 0.},         // row 3
        {0., 0., 0., 16960., 17610., -351.},  // row 4
        {0., 0., 0., 17610., 59120., -370.},  // row 5
        {0., 0., 0., -351., -370., 141470.}   // row 6
    });

    auto mass_matrix = gen_alpha_solver::create_matrix({
        {8.538e-2, 0., 0., 0., 0., 0.},    // row 1
        {0., 8.538e-2, 0., 0., 0., 0.},    // row 2
        {0., 0., 8.538e-2, 0., 0., 0.},    // row 3
        {0., 0., 0., 1.4433e-2, 0., 0.},   // row 4
        {0., 0., 0., 0., 0.40972e-2, 0.},  // row 5
        {0., 0., 0., 0., 0., 1.0336e-2}    // row 6
    });

    auto quadrature = UserDefinedQuadrature(
        {-0.9491079123427585, -0.7415311855993945, -0.4058451513773972, 0., 0.4058451513773972,
         0.7415311855993945, 0.9491079123427585},
        {0.1294849661688697, 0.2797053914892766, 0.3818300505051189, 0.4179591836734694,
         0.3818300505051189, 0.2797053914892766, 0.1294849661688697}
    );

    auto position_vectors =
        gen_alpha_solver::create_matrix({// node 1
                                         {0., 0., 0., 1., 0., 0., 0.},
                                         // node 2
                                         {1.7267316464601146, 0., 0., 1., 0., 0., 0.},
                                         // node 3
                                         {5., 0., 0., 1., 0., 0., 0.},
                                         // node 4
                                         {8.273268353539885, 0., 0., 1., 0., 0., 0.},
                                         // node 5
                                         {10., 0., 0., 1., 0., 0., 0.}});

    auto external_forces = std::vector<Forces*>{};
    auto constant_tip_force =
        GeneralizedForces(gen_alpha_solver::create_vector({0., 0., 1., 0., 0., 0.}), 5);
    external_forces.push_back(&constant_tip_force);

    std::shared_ptr<LinearizationParameters> dynamic_beam_lin_params =
        std::make_shared<DynamicBeamLinearizationParameters>(
            position_vectors, stiffness, mass_matrix, quadrature, external_forces
        );

    // Run the dynamic analysis for 1 iteration with a time step of 0.005 seconds
    auto time_stepper = gen_alpha_solver::TimeStepper(0., 0.005, 1, 10);

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

    // We expect the state to contain the following values after 0.005s via validation results
    // from BeamDyn at the tip node
    auto state_1 = results[1];
    auto position_1 = Kokkos::subview(
        state_1.generalized_coordinates, Kokkos::make_pair(4, 5), Kokkos::make_pair(0, 3)
    );
    openturbine::gen_alpha_solver::tests::expect_kokkos_view_2D_equal(
        position_1, {{-3.26341443E-09, -3.50249888E-08, 1.39513747E-04}}
    );
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
    auto velocity = gen_alpha_solver::create_matrix(
        {{0., 0., 0., 0., 0., 0.},  // node 1
         {0., 0., 0., 0., 0., 0.},  // node 2
         {0., 0., 0., 0., 0., 0.},  // node 3
         {0., 0., 0., 0., 0., 0.},  // node 4
         {0., 0., 0., 0., 0., 0.}}  // node 5
    );
    auto acceleration = gen_alpha_solver::create_matrix(
        {{0., 0., 0., 0., 0., 0.},  // node 1
         {0., 0., 0., 0., 0., 0.},  // node 2
         {0., 0., 0., 0., 0., 0.},  // node 3
         {0., 0., 0., 0., 0., 0.},  // node 4
         {0., 0., 0., 0., 0., 0.}}  // node 5
    );
    auto algo_acceleration = gen_alpha_solver::create_matrix(
        {{0., 0., 0., 0., 0., 0.},  // node 1
         {0., 0., 0., 0., 0., 0.},  // node 2
         {0., 0., 0., 0., 0., 0.},  // node 3
         {0., 0., 0., 0., 0., 0.},  // node 4
         {0., 0., 0., 0., 0., 0.}}  // node 5
    );

    auto initial_state = State{gen_coords, velocity, acceleration, algo_acceleration};

    // Set up the linearization parameters for the problem
    auto stiffness = gen_alpha_solver::create_matrix({
        {1368.17e3, 0., 0., 0., 0., 0.},      // row 1
        {0., 88560., 0., 0., 0., 0.},         // row 2
        {0., 0., 38780., 0., 0., 0.},         // row 3
        {0., 0., 0., 16960., 17610., -351.},  // row 4
        {0., 0., 0., 17610., 59120., -370.},  // row 5
        {0., 0., 0., -351., -370., 141470.}   // row 6
    });

    auto mass_matrix = gen_alpha_solver::create_matrix({
        {8.538e-2, 0., 0., 0., 0., 0.},    // row 1
        {0., 8.538e-2, 0., 0., 0., 0.},    // row 2
        {0., 0., 8.538e-2, 0., 0., 0.},    // row 3
        {0., 0., 0., 1.4433e-2, 0., 0.},   // row 4
        {0., 0., 0., 0., 0.40972e-2, 0.},  // row 5
        {0., 0., 0., 0., 0., 1.0336e-2}    // row 6
    });

    auto quadrature = UserDefinedQuadrature(
        {-0.9491079123427585, -0.7415311855993945, -0.4058451513773972, 0., 0.4058451513773972,
         0.7415311855993945, 0.9491079123427585},
        {0.1294849661688697, 0.2797053914892766, 0.3818300505051189, 0.4179591836734694,
         0.3818300505051189, 0.2797053914892766, 0.1294849661688697}
    );

    auto position_vectors =
        gen_alpha_solver::create_matrix({// node 1
                                         {0., 0., 0., 1., 0., 0., 0.},
                                         // node 2
                                         {1.7267316464601146, 0., 0., 1., 0., 0., 0.},
                                         // node 3
                                         {5., 0., 0., 1., 0., 0., 0.},
                                         // node 4
                                         {8.273268353539885, 0., 0., 1., 0., 0., 0.},
                                         // node 5
                                         {10., 0., 0., 1., 0., 0., 0.}});

    // Run the dynamic analysis for 3 iterations with a time step of 0.005 seconds
    auto time_stepper = gen_alpha_solver::TimeStepper(0., 0.005, 3, 10);

    auto external_forces = std::vector<Forces*>{};
    auto create_sin_varying_force = [](double t) {
        return gen_alpha_solver::create_vector({0., 0., 100. * std::sin(10. * t), 0., 0., 0.});
    };
    auto time_varying_force = TimeVaryingForces(create_sin_varying_force, 5);
    external_forces.push_back(&time_varying_force);

    std::shared_ptr<LinearizationParameters> dynamic_beam_lin_params =
        std::make_shared<DynamicBeamLinearizationParameters>(
            position_vectors, stiffness, mass_matrix, quadrature, external_forces
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

    // We expect the state to contain the following values after 0.005s via validation results
    // from BeamDyn at the tip node
    auto state_1 = results[1];
    auto position_1 = Kokkos::subview(
        state_1.generalized_coordinates, Kokkos::make_pair(4, 5), Kokkos::make_pair(0, 3)
    );
    openturbine::gen_alpha_solver::tests::expect_kokkos_view_2D_equal(
        position_1, {{-8.15173937E-08, -1.86549248E-07, 6.97278045E-04}}
    );

    // We expect the state to contain the following values after 0.010s via validation results
    // from BeamDyn at the tip node
    auto state_2 = results[2];
    auto position_2 = Kokkos::subview(
        state_2.generalized_coordinates, Kokkos::make_pair(4, 5), Kokkos::make_pair(0, 3)
    );
    openturbine::gen_alpha_solver::tests::expect_kokkos_view_2D_equal(
        position_2, {{-1.00926258E-06, -7.91711079E-07, 2.65017558E-03}}
    );

    // We expect the state to contain the following values after 0.015s via validation results
    // from BeamDyn at the tip node
    auto state_3 = results[3];
    auto position_3 = Kokkos::subview(
        state_3.generalized_coordinates, Kokkos::make_pair(4, 5), Kokkos::make_pair(0, 3)
    );
    openturbine::gen_alpha_solver::tests::expect_kokkos_view_2D_equal(
        position_3, {{-5.05830945E-06, -2.29457246E-06, 6.30508154E-03}}
    );
}

}  // namespace openturbine::gebt_poc::tests
