#include <fstream>
#include <initializer_list>
#include <iostream>

#include <gtest/gtest.h>

#include "src/restruct_poc/BeamElement.hpp"
#include "src/restruct_poc/BeamNode.hpp"
#include "src/restruct_poc/BeamSection.hpp"
#include "src/restruct_poc/BeamsInput.hpp"
#include "src/restruct_poc/CreateBeams.hpp"
#include "src/restruct_poc/beams.hpp"
#include "src/restruct_poc/solver/InitializeConstraints.hpp"
#include "src/restruct_poc/solver/Solver.hpp"
#include "src/restruct_poc/solver/Step.hpp"
#include "tests/unit_tests/gen_alpha_poc/test_utilities.h"

using BeamQuadrature = std::vector<std::array<double, 2>>;
using Array_6x6 = std::array<std::array<double, 6>, 6>;
using Array_7 = std::array<double, 7>;

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
std::vector<double> node_s(
    {0., 0.11747233803526763, 0.35738424175967748, 0.64261575824032247, 0.88252766196473242, 1.}
);

// Element quadrature
BeamQuadrature quadrature{
    {-0.9491079123427585, 0.1294849661688697},  {-0.7415311855993943, 0.27970539148927664},
    {-0.40584515137739696, 0.3818300505051189}, {6.123233995736766e-17, 0.4179591836734694},
    {0.4058451513773971, 0.3818300505051189},   {0.7415311855993945, 0.27970539148927664},
    {0.9491079123427585, 0.1294849661688697},
};

std::vector<BeamSection> sections = {
    BeamSection(0., mass_matrix, stiffness_matrix),
    BeamSection(1., mass_matrix, stiffness_matrix),
};

TEST(RotatingBeamTest, StepConvergence) {
    // Gravity vector
    std::array<double, 3> gravity = {0., 0., 0.};

    // Build vector of nodes (straight along x axis, no rotation)
    // Calculate displacement, velocity, acceleration assuming a
    // 0.1 rad/s angular velocity around the z axis
    const double omega = 0.1;
    std::vector<BeamNode> nodes;
    std::vector<std::array<double, 7>> displacement;
    std::vector<std::array<double, 6>> velocity;
    std::vector<std::array<double, 6>> acceleration;
    for (const double s : node_s) {
        auto x = 10 * s + 2.;
        nodes.push_back(BeamNode(s, {x, 0., 0., 1., 0., 0., 0.}));
        displacement.push_back({0., 0., 0., 1., 0., 0., 0.});
        velocity.push_back({0., x * omega, 0., 0., 0., omega});
        acceleration.push_back({0., 0., 0., 0., 0., 0.});
    }

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
    const int num_system_nodes(beams.num_nodes);

    // Constraint inputs
    std::vector<ConstraintInput> constraint_inputs({ConstraintInput(-1, 0)});

    // Solution parameters
    const bool is_dynamic_solve(true);
    const int max_iter(5);
    const double step_size(0.01);  // seconds
    const double rho_inf(0.9);

    // Create solver
    Solver solver(
        is_dynamic_solve, max_iter, step_size, rho_inf, num_system_nodes, constraint_inputs,
        displacement, velocity, acceleration
    );

    // Initialize constraints
    InitializeConstraints(solver, beams);

    // Perform 10 time steps and check for convergence within max_iter iterations
    for (int i = 0; i < 10; ++i) {
        // Set constraint displacement
        auto q = openturbine::gen_alpha_solver::quaternion_from_rotation_vector(
            Vector(0, 0, omega * step_size * (i + 1))
        );
        solver.constraints.UpdateDisplacement(
            0, {0, 0, 0, q.GetScalarComponent(), q.GetXComponent(), q.GetYComponent(),
                q.GetZComponent()}
        );
        auto converged = Step(solver, beams);
        EXPECT_EQ(converged, true);
    }

    openturbine::gen_alpha_solver::tests::expect_kokkos_view_2D_equal(
        solver.state.q,
        {
            {-0.000099661884299369481, 0.019999672917628962, -3.6608854058480302E-25,
             0.99998750002604175, -1.5971376141505654E-26, 3.1592454262792375E-25,
             0.004999979166692714},
            {-0.00015838391157346692, 0.031746709275713193, -2.8155520815870626E-13,
             0.99998750002143066, 2.7244338869052949E-12, 1.989181042516661E-12,
             0.0049999800888738608},
            {-0.00027859681974392133, 0.055737500699772298, 2.815269319303426E-12,
             0.9999875000205457, 7.3510877107173739E-12, 1.0550370096863904E-12,
             0.0049999802658924715},
            {-0.00042131446700509681, 0.08426017738413949, 8.2854411551089936E-12,
             0.99998750002161218, 3.7252296525466957E-11, -5.26890056047209E-14,
             0.0049999800525935617},
            {-0.00054093210652801399, 0.10825097509997549, -9.3934322245617647E-12,
             0.99998750002142056, 4.0321076018153484E-11, 5.2579938812420674E-12,
             0.0049999800909203019},
            {-0.00059944528351138049, 0.11999801747595988, -2.6207280972097857E-11,
             0.99998750002237801, 3.4435006114567926E-11, 6.4250095159262128E-12,
             0.0049999798994432168},
        }
    );
}

TEST(RotatingBeamTest, TwoBeam) {
    // Gravity vector
    std::array<double, 3> gravity = {0., 0., 0.};

    // Rotor angular velocity in rad/s
    Vector omega(0., 0., 1);

    // Build vector of nodes (straight along x axis, no rotation)
    // Calculate displacement, velocity, acceleration assuming a
    // 0.1 rad/s angular velocity around the z axis
    const int num_blades = 2;
    std::vector<BeamElement> blade_elems;
    std::vector<std::array<double, 7>> displacement;
    std::vector<std::array<double, 6>> velocity;
    std::vector<std::array<double, 6>> acceleration;
    std::vector<ConstraintInput> constraint_inputs;
    // Loop through blades
    for (int i = 0; i < num_blades; ++i) {
        // Define root rotation
        Quaternion q_root(1, 0, 0, 0);

        // Declare list of element nodes
        std::vector<BeamNode> nodes;

        // Loop through nodes
        for (const double s : node_s) {
            auto pos = q_root * Vector(10. * s + 2., 0, 0);
            nodes.push_back(BeamNode(s, pos, q_root));

            // Add node initial displacement, velocity, and acceleration
            displacement.push_back({0., 0., 0., 1., 0., 0., 0.});
            auto v = omega.CrossProduct(pos);
            velocity.push_back({
                v.GetX(),
                v.GetY(),
                v.GetZ(),
                omega.GetXComponent(),
                omega.GetYComponent(),
                omega.GetZComponent(),
            });
            acceleration.push_back({0., 0., 0., 0., 0., 0.});
        }

        // Add beam element
        blade_elems.push_back(BeamElement(nodes, sections, quadrature));

        // Set constraint nodes
        constraint_inputs.push_back(ConstraintInput(-1, i * node_s.size()));
    }

    // Define beam initialization
    BeamsInput beams_input(blade_elems, gravity);

    // Initialize beams from element inputs
    auto beams = CreateBeams(beams_input);

    // Number of system nodes from number of beam nodes
    const int num_system_nodes(beams.num_nodes);

    // Solution parameters
    const bool is_dynamic_solve(true);
    const int max_iter(1);
    const double step_size(0.01);  // seconds
    const double rho_inf(0.9);

    // Create solver with initial node state
    Solver solver(
        is_dynamic_solve, max_iter, step_size, rho_inf, num_system_nodes, constraint_inputs,
        displacement, velocity, acceleration
    );

    // Initialize constraints
    InitializeConstraints(solver, beams);

    // Calculate hub rotation for this time step
    auto q_hub = openturbine::gen_alpha_solver::quaternion_from_rotation_vector(omega * step_size);

    // Define hub translation/rotation displacement
    Array_7 u_hub(
        {0, 0, 0, q_hub.GetScalarComponent(), q_hub.GetXComponent(), q_hub.GetYComponent(),
         q_hub.GetZComponent()}
    );

    // Update constraint displacements
    for (int j = 0; j < solver.num_constraint_nodes; ++j) {
        solver.constraints.UpdateDisplacement(j, u_hub);
    }

    // Take step, don't check for convergence, the following tests check that
    // all the elements were assembled properly
    Step(solver, beams);

    auto n = solver.num_system_dofs / 2;
    auto m = solver.num_constraint_dofs / 2;

    // Check that K matrix is the same for both beams
    auto K = openturbine::gen_alpha_solver::tests::kokkos_view_2D_to_vector(solver.K);
    // WriteMatrixToFile(K, "K.csv");
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            EXPECT_NEAR(K[i][j], K[n + i][n + j], 1.e-10);
        }
    }

    // Check that M matrix is the same for both beams
    auto M = openturbine::gen_alpha_solver::tests::kokkos_view_2D_to_vector(solver.M);
    // WriteMatrixToFile(M, "M.csv");
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            EXPECT_NEAR(M[i][j], M[n + i][n + j], 1.e-10);
        }
    }

    // Check that St matrix is the same for both beams
    auto St = openturbine::gen_alpha_solver::tests::kokkos_view_2D_to_vector(solver.St);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            EXPECT_NEAR(St[i][j], St[n + i][n + j], 1.e-10);
        }
    }

    // Check that R vector is the same for both beams
    auto R = openturbine::gen_alpha_solver::tests::kokkos_view_1D_to_vector(solver.R);
    for (int i = 0; i < n; ++i) {
        EXPECT_NEAR(R[i], R[n + i], 1.e-10);
    }

    // Check that Phi vector is the same for both beams
    auto Phi =
        openturbine::gen_alpha_solver::tests::kokkos_view_1D_to_vector(solver.constraints.Phi);
    for (int i = 0; i < m; ++i) {
        EXPECT_NEAR(Phi[i], Phi[i + m], 1.e-10);
    }

    // Check that B matrix is the same for both beams
    auto B = openturbine::gen_alpha_solver::tests::kokkos_view_2D_to_vector(solver.constraints.B);
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            EXPECT_NEAR(B[i][j], B[i + m][j + n], 1.e-10);
        }
    }
}

TEST(RotatingBeamTest, ThreeBladeRotor) {
    // Gravity vector
    std::array<double, 3> gravity = {0., 0., 9.81};

    // Rotor angular velocity in rad/s
    Vector omega(0., 0., 1);

    // Build vector of nodes (straight along x axis, no rotation)
    // Calculate displacement, velocity, acceleration assuming a
    // 1 rad/s angular velocity around the z axis
    const int num_blades = 3;
    std::vector<BeamElement> blade_elems;
    std::vector<std::array<double, 7>> displacement;
    std::vector<std::array<double, 6>> velocity;
    std::vector<std::array<double, 6>> acceleration;
    std::vector<ConstraintInput> constraint_inputs;

    // Loop through blades
    for (int i = 0; i < num_blades; ++i) {
        // Define root rotation
        auto q_root = openturbine::gen_alpha_solver::quaternion_from_rotation_vector(
            Vector(0, 0, 2.0 * M_PI * i / num_blades)
        );

        // Declare list of element nodes
        std::vector<BeamNode> nodes;

        // Loop through nodes
        for (const double s : node_s) {
            auto pos = q_root * Vector(10. * s + 2., 0, 0);
            nodes.push_back(BeamNode(s, pos, q_root));

            // Add node initial displacement, velocity, and acceleration
            displacement.push_back({0., 0., 0., 1., 0., 0., 0.});
            auto v = omega.CrossProduct(pos);
            velocity.push_back({
                v.GetX(),
                v.GetY(),
                v.GetZ(),
                omega.GetXComponent(),
                omega.GetYComponent(),
                omega.GetZComponent(),
            });
            acceleration.push_back({0., 0., 0., 0., 0., 0.});
        }

        // Add beam element
        blade_elems.push_back(BeamElement(nodes, sections, quadrature));

        // Set constraint nodes
        constraint_inputs.push_back(ConstraintInput(-1, i * node_s.size()));
    }

    // Define beam initialization
    BeamsInput beams_input(blade_elems, gravity);

    // Initialize beams from element inputs
    auto beams = CreateBeams(beams_input);

    // Number of system nodes from number of beam nodes
    const int num_system_nodes(beams.num_nodes);

    // Solution parameters
    const bool is_dynamic_solve(true);
    const int max_iter(4);
    const double step_size(0.01);  // seconds
    const double rho_inf(0.9);
    const double t_end(0.1);
    const int num_steps(t_end / step_size + 1.0);

    // Create solver with initial node state
    Solver solver(
        is_dynamic_solve, max_iter, step_size, rho_inf, num_system_nodes, constraint_inputs,
        displacement, velocity, acceleration
    );

    // Initialize constraints
    InitializeConstraints(solver, beams);

    // Perform time steps and check for convergence within max_iter iterations
    for (int i = 0; i < num_steps; ++i) {
        // Calculate hub rotation for this time step
        auto q_hub = openturbine::gen_alpha_solver::quaternion_from_rotation_vector(
            omega * step_size * (i + 1)
        );

        // Define hub translation/rotation displacement
        Array_7 u_hub(
            {0, 0, 0, q_hub.GetScalarComponent(), q_hub.GetXComponent(), q_hub.GetYComponent(),
             q_hub.GetZComponent()}
        );

        // Update constraint displacements
        for (int j = 0; j < solver.num_constraint_nodes; ++j) {
            solver.constraints.UpdateDisplacement(j, u_hub);
        }

        // Take step
        auto converged = Step(solver, beams);

        // Verify that step converged
        EXPECT_EQ(converged, true);
    }
}

}  // namespace openturbine::restruct_poc::tests
