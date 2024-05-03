#include <fstream>
#include <initializer_list>
#include <iostream>

#include <gtest/gtest.h>

#include "src/restruct_poc/beams/beam_element.hpp"
#include "src/restruct_poc/beams/beam_node.hpp"
#include "src/restruct_poc/beams/beam_section.hpp"
#include "src/restruct_poc/beams/beams.hpp"
#include "src/restruct_poc/beams/beams_input.hpp"
#include "src/restruct_poc/beams/create_beams.hpp"
#include "src/restruct_poc/solver/initialize_constraints.hpp"
#include "src/restruct_poc/solver/solver.hpp"
#include "src/restruct_poc/solver/step.hpp"
#include "tests/unit_tests/gen_alpha_poc/test_utilities.h"

using BeamQuadrature = std::vector<std::array<double, 2>>;
using Array_6x6 = std::array<std::array<double, 6>, 6>;

namespace openturbine::restruct_poc::tests {

template <typename T>
void WriteMatrixToFile(const std::vector<std::vector<T>>& data, const std::string& filename) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Unable to open file: " << filename << std::endl;
        return;
    }
    for (const auto& innerVector : data) {
        for (const auto& element : innerVector) {
            file << element << ",";
        }
        file << std::endl;
    }
    file.close();
}

TEST(DynamicBeamTest, CantileverBeamSineLoad) {
    // Mass matrix for uniform composite beam section
    Array_6x6 mass_matrix = {{
        {8.538e-2, 0., 0., 0., 0., 0.},
        {0., 8.538e-2, 0., 0., 0., 0.},
        {0., 0., 8.538e-2, 0., 0., 0.},
        {0., 0., 0., 1.4433e-2, 0., 0.},
        {0., 0., 0., 0., 0.40972e-2, 0.},
        {0., 0., 0., 0., 0., 1.0336e-2},
    }};

    // Stiffness matrix for uniform composite beam section
    Array_6x6 stiffness_matrix = {{
        {1368.17e3, 0., 0., 0., 0., 0.},
        {0., 88.56e3, 0., 0., 0., 0.},
        {0., 0., 38.78e3, 0., 0., 0.},
        {0., 0., 0., 16.9600e3, 17.6100e3, -0.3510e3},
        {0., 0., 0., 17.6100e3, 59.1200e3, -0.3700e3},
        {0., 0., 0., -0.3510e3, -0.3700e3, 141.470e3},
    }};

    // Node locations (GLL quadrature)
    std::vector<double> node_s({0., 0.17267316464601146, 0.5, 0.8273268353539885, 1.});

    // Element quadrature
    BeamQuadrature quadrature{
        {-0.9491079123427585, 0.1294849661688697},  {-0.7415311855993943, 0.27970539148927664},
        {-0.40584515137739696, 0.3818300505051189}, {6.123233995736766e-17, 0.4179591836734694},
        {0.4058451513773971, 0.3818300505051189},   {0.7415311855993945, 0.27970539148927664},
        {0.9491079123427585, 0.1294849661688697},
    };
    std::cout << 1 << std::endl;
    std::vector<BeamSection> sections = {
        BeamSection(0., mass_matrix, stiffness_matrix),
        BeamSection(1., mass_matrix, stiffness_matrix),
    };

    // Gravity vector
    std::array<double, 3> gravity = {0., 0., 0.};

    // Build vector of nodes (straight along x axis, no rotation)
    std::vector<BeamNode> nodes;
    std::vector<std::array<double, 7>> displacement;
    std::vector<std::array<double, 6>> velocity;
    std::vector<std::array<double, 6>> acceleration;
    for (const double s : node_s) {
        nodes.push_back(BeamNode(s, {10 * s, 0., 0., 1., 0., 0., 0.}));
        displacement.push_back({0., 0., 0., 1., 0., 0., 0.});
        velocity.push_back({0., 0., 0., 0., 0., 0.});
        acceleration.push_back({0., 0., 0., 0., 0., 0.});
    }

    std::cout << 2 << std::endl;
    // Define beam initialization
    BeamsInput beams_input(
        {
            BeamElement(nodes, sections, quadrature),
        },
        gravity
    );

    // Initialize beams from element inputs
    auto beams = CreateBeams(beams_input);

    // Number of system nodes from number of beam nodes
    const size_t num_system_nodes(beams.num_nodes);

    // Constraint inputs
    std::vector<ConstraintInput> constraint_inputs({ConstraintInput(-1, 0)});

    std::cout << 3 << std::endl;
    // Solution parameters
    const bool is_dynamic_solve(true);
    const size_t max_iter(5);
    const double step_size(0.005);  // seconds
    const double rho_inf(0.0);

    // Create solver
    Solver solver(
        is_dynamic_solve, max_iter, step_size, rho_inf, num_system_nodes, constraint_inputs,
        displacement, velocity, acceleration
    );

    std::cout << 4 << std::endl;
    // Initialize constraints
    InitializeConstraints(solver, beams);
    solver.constraints.UpdateDisplacement(0, {0, 0, 0, 1, 0, 0, 0});

    std::cout << 5 << std::endl;
    // First step
    Kokkos::deep_copy(
        Kokkos::subview(beams.node_FX, beams.num_nodes - 1, 2), 100. * std::sin(10.0 * 0.005)
    );
    std::cout << 6 << std::endl;
    auto converged = Step(solver, beams);
    EXPECT_EQ(converged, true);
    std::cout << 7 << std::endl;
    {
        Kokkos::View<double[3]> result("result");
        Kokkos::deep_copy(result, Kokkos::subview(solver.state.q, 4, Kokkos::make_pair(0, 3)));
        openturbine::gen_alpha_solver::tests::expect_kokkos_view_1D_equal(
            result, {-8.15173937E-08, -1.86549248E-07, 6.97278045E-04}
        );
    }
    std::cout << 8 << std::endl;
    // Second step
    Kokkos::deep_copy(
        Kokkos::subview(beams.node_FX, beams.num_nodes - 1, 2), 100. * std::sin(10.0 * 0.010)
    );
    std::cout << 9 << std::endl;
    converged = Step(solver, beams);
    EXPECT_EQ(converged, true);
    std::cout << 10 << std::endl;
    {
        Kokkos::View<double[3]> result("result");
        Kokkos::deep_copy(result, Kokkos::subview(solver.state.q, 4, Kokkos::make_pair(0, 3)));
        openturbine::gen_alpha_solver::tests::expect_kokkos_view_1D_equal(
            result, {-1.00926258E-06, -7.91711079E-07, 2.65017558E-03}
        );
    }
    std::cout << 11 << std::endl;

    // Third step
    Kokkos::deep_copy(
        Kokkos::subview(beams.node_FX, beams.num_nodes - 1, 2), 100. * std::sin(10.0 * 0.015)
    );
    std::cout << 12 << std::endl;
    converged = Step(solver, beams);
    EXPECT_EQ(converged, true);
    std::cout << 13 << std::endl;
    {
        Kokkos::View<double[3]> result("result");
        Kokkos::deep_copy(result, Kokkos::subview(solver.state.q, 4, Kokkos::make_pair(0, 3)));
        openturbine::gen_alpha_solver::tests::expect_kokkos_view_1D_equal(
            result, {-5.05830945E-06, -2.29457246E-06, 6.30508154E-03}
        );
    }
    std::cout << 14 << std::endl;
}

}  // namespace openturbine::restruct_poc::tests
