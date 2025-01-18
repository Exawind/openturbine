#include <gtest/gtest.h>

#include "dof_management/assemble_node_freedom_allocation_table.hpp"
#include "dof_management/compute_node_freedom_map_table.hpp"
#include "dof_management/create_constraint_freedom_table.hpp"
#include "dof_management/create_element_freedom_table.hpp"
#include "elements/beams/create_beams.hpp"
#include "elements/elements.hpp"
#include "elements/masses/create_masses.hpp"
#include "elements/springs/create_springs.hpp"
#include "elements/springs/springs.hpp"
#include "model/model.hpp"
#include "solver/solver.hpp"
#include "step/step.hpp"
#include "test_utilities.hpp"

namespace openturbine::tests {

/*
 * A simple spring-mass system with one mass attached to a fixed point by a spring.
 * The mass oscillates horizontally with simple harmonic motion. The system is initialized
 * with the mass displaced 2 units from equilibrium position.
 *
 * Schematic of the system with mass (M) and spring (/\/\/), fixed BC at left end node displacement
 * applied at right node:
 *
 * /|
 * /|o---/\/\/--- ( M )o---->
 * /|
 */
inline auto SetUpSpringMassSystem() {
    auto model = Model();

    // Add two nodes for the spring element
    const auto fixed_node_id =
        model.AddNode().SetPosition(0., 0., 0., 1., 0., 0., 0.).Build();  // First node at origin
    const auto mass_node_id =
        model.AddNode().SetPosition(2., 0., 0., 1., 0., 0., 0.).Build();  // Second node at (2,0,0)

    // Add mass element
    constexpr auto m = 1.;
    constexpr auto j = 1.;
    model.AddMassElement(
        mass_node_id, {{
                          {m, 0., 0., 0., 0., 0.},  // mass in x-direction
                          {0., m, 0., 0., 0., 0.},  // mass in y-direction
                          {0., 0., m, 0., 0., 0.},  // mass in z-direction
                          {0., 0., 0., j, 0., 0.},  // inertia around x-axis
                          {0., 0., 0., 0., j, 0.},  // inertia around y-axis
                          {0., 0., 0., 0., 0., j},  // inertia around z-axis
                      }}
    );

    // Add fixed BC to the first node
    model.AddFixedBC(fixed_node_id);

    // Add spring element
    const auto k = 10.;  // stiffness
    const auto l0 = 0.;  // undeformed length
    model.AddSpringElement(fixed_node_id, mass_node_id, k, l0);

    // The spring-mass system should move periodically between -2 and 2 (position of the second node)
    // with simple harmonic motion where the time period is 2 * pi * sqrt(m / k)
    const double T = 2. * M_PI * sqrt(m / k);
    constexpr auto num_steps = 1000;

    // Set up step parameters
    constexpr bool is_dynamic_solve(true);
    constexpr size_t max_iter(6);
    constexpr double rho_inf(0.);                                // No damping
    const double step_size(T / static_cast<double>(num_steps));  // Calculate step size
    auto parameters = StepParameters(is_dynamic_solve, max_iter, step_size, rho_inf);

    // Create solver, elements, constraints, and state
    auto [state, elements, constraints, solver] = model.CreateSystemWithSolver();

    // Create host mirror for checking solution
    auto q = Kokkos::create_mirror(state.q);

    // Run simulation for T seconds
    for (auto time_step = 1U; time_step <= num_steps; ++time_step) {
        auto converged = Step(parameters, solver, elements, state, constraints);
        EXPECT_TRUE(converged);
        // Simulation at time T / 2
        if (time_step == num_steps / 2) {
            Kokkos::deep_copy(q, state.q);
            EXPECT_EQ(q(0, 0), 0.);  // First node is fixed
            EXPECT_NEAR(
                q(1, 0), -3.9999199193098396, 1.e-12
            );  // Second node should have displacement close to -4.0 after T/2
        }
        // Simulation at time T
        if (time_step == num_steps) {
            Kokkos::deep_copy(q, state.q);
            EXPECT_EQ(q(0, 0), 0.);  // First node is fixed
            EXPECT_NEAR(
                q(1, 0), -8.1226588438437419e-05, 1.e-12
            );  // Second node should have displacement close to 0. after T
        }
    }
}

TEST(SpringMassSystemTest, FinalDisplacement) {
    SetUpSpringMassSystem();
}

/*
 * A chain of identical masses held together by identical springs and anchored at both ends.
 * When the masses are equidistant from eachother, the system shoud stay in perfect static
 * equilibrium. By adjusting the variable number_of_masses, the size of the system can be scaled up
 * or down.
 *
 * ASCII Art of the system with three masses (M) and anchor points (A):
 *
 * A ---/\/\/--- M ---/\/\/--- M ---/\/\/--- M ---/\/\/--- A
 */
inline auto SetUpSpringMassChainSystem() {
    auto model = Model();

    // Add nodes for each mass and an anchor point on each side
    constexpr auto number_of_masses = 10U;
    constexpr auto displacement = 0.5;
    auto position = 0.;
    model.AddNode().SetPosition(position, 0., 0., 1., 0., 0., 0.);
    for (auto mass_number = 0U; mass_number < number_of_masses; ++mass_number) {
        position += displacement;
        model.AddNode().SetPosition(position, 0., 0., 1., 0., 0., 0.);
    }
    position += displacement;
    model.AddNode().SetPosition(position, 0., 0., 1., 0., 0., 0.);

    // Mass matrix (Identical for all masses)
    constexpr auto m = 1.;
    constexpr auto j = 1.;
    constexpr auto mass_matrix = std::array{
        std::array{m, 0., 0., 0., 0., 0.},  // mass in x-direction
        std::array{0., m, 0., 0., 0., 0.},  // mass in y-direction
        std::array{0., 0., m, 0., 0., 0.},  // mass in z-direction
        std::array{0., 0., 0., j, 0., 0.},  // inertia around x-axis
        std::array{0., 0., 0., 0., j, 0.},  // inertia around y-axis
        std::array{0., 0., 0., 0., 0., j},  // inertia around z-axis
    };

    for (auto mass_number = 0U; mass_number < number_of_masses; ++mass_number) {
        model.AddMassElement(mass_number + 1, {mass_matrix});
    }

    // Create springs
    const auto k = 10.;
    for (auto mass_number = 0U; mass_number <= number_of_masses; ++mass_number) {
        model.AddSpringElement(mass_number, mass_number + 1, k, 0.);
    }

    // Add fixed BC to the anchor nodes
    model.AddFixedBC(0);
    model.AddFixedBC(number_of_masses + 1);

    // Create system and solver
    auto [state, elements, constraints] = model.CreateSystem();
    auto solver = CreateSolver(state, elements, constraints);

    const double T = 2. * M_PI * sqrt(m / k);
    constexpr auto num_steps = 1000;

    // Set up step parameters
    constexpr bool is_dynamic_solve(true);
    constexpr size_t max_iter(6);
    constexpr double rho_inf(0.);                                // No damping
    const double step_size(T / static_cast<double>(num_steps));  // Calculate step size
    auto parameters = StepParameters(is_dynamic_solve, max_iter, step_size, rho_inf);

    auto q = Kokkos::create_mirror(state.q);

    // Run simulation for T seconds
    for (auto time_step = 1U; time_step <= num_steps; ++time_step) {
        auto converged = Step(parameters, solver, elements, state, constraints);
        EXPECT_TRUE(converged);
        Kokkos::deep_copy(q, state.q);
        for (auto node = 0U; node < q.extent(0); ++node) {
            EXPECT_EQ(q(node, 0), 0.);
        }
    }
}

TEST(SpringMassChainSystemTest, FinalDisplacement) {
    SetUpSpringMassChainSystem();
}

}  // namespace openturbine::tests
