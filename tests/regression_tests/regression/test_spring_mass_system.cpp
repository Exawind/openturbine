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
    auto solver = CreateSolver(state, elements, constraints);

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
    model.AddNode(
        {position, 0., 0., 1., 0., 0., 0.},  // Left anchor point
        {0., 0., 0., 1., 0., 0., 0.},        // initial displacement
        {0., 0., 0., 0., 0., 0.},            // initial velocity
        {0., 0., 0., 0., 0., 0.}             // initial acceleration
    );
    for (auto mass_number = 0U; mass_number < number_of_masses; ++mass_number) {
        position += displacement;
        model.AddNode(
            {position, 0., 0., 1., 0., 0., 0.},  // Mass location
            {0., 0., 0., 1., 0., 0., 0.},        // initial displacement
            {0., 0., 0., 0., 0., 0.},            // initial velocity
            {0., 0., 0., 0., 0., 0.}             // initial acceleration
        );
    }
    position += displacement;
    model.AddNode(
        {position, 0., 0., 1., 0., 0., 0.},  // Right anchor point
        {0., 0., 0., 1., 0., 0., 0.},        // initial displacement
        {0., 0., 0., 0., 0., 0.},            // initial velocity
        {0., 0., 0., 0., 0., 0.}             // initial acceleration
    );

    // No beams
    const auto beams_input = BeamsInput({}, {0., 0., 0.});
    auto beams = CreateBeams(beams_input);

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

    auto mass_elements = std::vector<MassElement>{};
    for (auto mass_number = 0U; mass_number < number_of_masses; ++mass_number) {
        mass_elements.emplace_back(model.GetNode(mass_number + 1), mass_matrix);
    }
    const auto masses_input = MassesInput(mass_elements, {0., 0., 0.});
    auto masses = CreateMasses(masses_input);

    // Create springs
    const auto k = 10.;
    auto spring_elements = std::vector<SpringElement>{};
    for (auto mass_number = 0U; mass_number <= number_of_masses; ++mass_number) {
        spring_elements.emplace_back(
            std::array{model.GetNode(mass_number), model.GetNode(mass_number + 1)}, k, 0.
        );
    }
    const auto springs_input = SpringsInput(spring_elements);
    auto springs = CreateSprings(springs_input);

    // Create elements
    auto elements = Elements{beams, masses, springs};

    // Add fixed BC to the anchor nodes
    model.AddFixedBC(model.GetNode(0));
    model.AddFixedBC(model.GetNode(number_of_masses + 1));

    // Set up solver components
    auto state = model.CreateState();
    auto constraints = Constraints(model.GetConstraints());
    assemble_node_freedom_allocation_table(state, elements, constraints);
    compute_node_freedom_map_table(state);
    create_element_freedom_table(elements, state);
    create_constraint_freedom_table(constraints, state);
    auto solver = Solver(
        state.ID, state.node_freedom_allocation_table, state.node_freedom_map_table,
        elements.NumberOfNodesPerElement(), elements.NodeStateIndices(), constraints.num_dofs,
        constraints.type, constraints.base_node_freedom_table, constraints.target_node_freedom_table,
        constraints.row_range
    );

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
