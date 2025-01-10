#include <gtest/gtest.h>

#include "test_utilities.hpp"

#include "src/dof_management/assemble_node_freedom_allocation_table.hpp"
#include "src/dof_management/compute_node_freedom_map_table.hpp"
#include "src/dof_management/create_constraint_freedom_table.hpp"
#include "src/dof_management/create_element_freedom_table.hpp"
#include "src/elements/beams/create_beams.hpp"
#include "src/elements/elements.hpp"
#include "src/elements/masses/create_masses.hpp"
#include "src/elements/springs/create_springs.hpp"
#include "src/elements/springs/springs.hpp"
#include "src/model/model.hpp"
#include "src/solver/solver.hpp"
#include "src/step/step.hpp"

namespace openturbine::tests {

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
    auto state = model.CreateState();
    auto elements = model.CreateElements();
    auto constraints = model.CreateConstraints();
    auto solver = Model::CreateSolver(state, elements, constraints);

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
                q(1, 0), -3.9999200547674469, 1.e-12
            );  // Second node should have displacement close to -4.0 after T/2
        }
        // Simulation at time T
        if (time_step == num_steps) {
            Kokkos::deep_copy(q, state.q);
            EXPECT_EQ(q(0, 0), 0.);  // First node is fixed
            EXPECT_NEAR(
                q(1, 0), -8.0949125285126051e-05, 1.e-12
            );  // Second node should have displacement close to 0. after T
        }
    }
}

TEST(SpringMassSystemTest, FinalDisplacement) {
    SetUpSpringMassSystem();
}

}  // namespace openturbine::tests
