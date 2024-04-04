#include <initializer_list>

#include <gtest/gtest.h>

#include "src/restruct_poc/beams.hpp"
#include "src/restruct_poc/solver.hpp"
#include "tests/unit_tests/gen_alpha_poc/test_utilities.h"

namespace oturb::restruct_poc::tests {

TEST(RotatingBeamTest, StepConvergence) {
    // Mass matrix for uniform composite beam section
    std::array<std::array<double, 6>, 6> mass_matrix = {{
        {8.538e-2, 0., 0., 0., 0., 0.},
        {0., 8.538e-2, 0., 0., 0., 0.},
        {0., 0., 8.538e-2, 0., 0., 0.},
        {0., 0., 0., 1.4433e-2, 0., 0.},
        {0., 0., 0., 0., 0.40972e-2, 0.},
        {0., 0., 0., 0., 0., 1.0336e-2},
    }};

    // Stiffness matrix for uniform composite beam section
    std::array<std::array<double, 6>, 6> stiffness_matrix = {{
        {1368.17e3, 0., 0., 0., 0., 0.},
        {0., 88.56e3, 0., 0., 0., 0.},
        {0., 0., 38.78e3, 0., 0., 0.},
        {0., 0., 0., 16.9600e3, 17.6100e3, -0.3510e3},
        {0., 0., 0., 17.6100e3, 59.1200e3, -0.3700e3},
        {0., 0., 0., -0.3510e3, -0.3700e3, 141.470e3},
    }};

    // Gravity vector
    std::array<double, 3> gravity = {0., 0., 0.};

    // Node locations (GLL quadrature)
    std::vector<double> node_s(
        {0., 0.11747233803526763, 0.35738424175967748, 0.64261575824032247, 0.88252766196473242, 1.}
    );

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
            BeamElement(
                nodes,
                {
                    BeamSection(0., mass_matrix, stiffness_matrix),
                    BeamSection(1., mass_matrix, stiffness_matrix),
                },
                BeamQuadrature{
                    {-0.9491079123427585, 0.1294849661688697},
                    {-0.7415311855993943, 0.27970539148927664},
                    {-0.40584515137739696, 0.3818300505051189},
                    {6.123233995736766e-17, 0.4179591836734694},
                    {0.4058451513773971, 0.3818300505051189},
                    {0.7415311855993945, 0.27970539148927664},
                    {0.9491079123427585, 0.1294849661688697},
                },
                {0., 0., 0., 1., 0., 0., 0}  // Root node position and rotation
            ),
        },
        gravity
    );

    // Initialize beams from element inputs
    auto beams = CreateBeams(beams_input);

    // Number of system nodes from number of beam nodes
    const size_t num_system_nodes(beams.num_nodes);

    // Constraint inputs
    std::vector<ConstraintInput> constraint_inputs({ConstraintInput(-1, 0)});

    // Solution parameters
    const bool is_dynamic_solve(true);
    const size_t max_iter(3);
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
    for (size_t i = 0; i < 10; ++i) {
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
}

}  // namespace oturb::restruct_poc::tests
