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
    model.AddNode(
        {0., 0., 0., 1., 0., 0., 0.},  // First node at origin
        {0., 0., 0., 1., 0., 0., 0.},  //
        {0., 0., 0., 0., 0., 0.},      //
        {0., 0., 0., 0., 0., 0.}
    );
    model.AddNode(
        {2., 0., 0., 1., 0., 0., 0.},  // Second node at (2,0,0)
        {0., 0., 0., 1., 0., 0., 0.},  //
        {0., 0., 0., 0., 0., 0.},      //
        {0., 0., 0., 0., 0., 0.}
    );

    // No beams
    const auto beams_input = BeamsInput({}, {0., 0., 0.});
    auto beams = CreateBeams(beams_input);

    // We need to add a mass element with identity for mass matrix to create a spring-mass system
    constexpr auto m = 1.;
    constexpr auto j = 1.;
    constexpr auto mass_matrix = std::array{
        std::array{m, 0., 0., 0., 0., 0.},  //
        std::array{0., m, 0., 0., 0., 0.},  //
        std::array{0., 0., m, 0., 0., 0.},  //
        std::array{0., 0., 0., j, 0., 0.},  //
        std::array{0., 0., 0., 0., j, 0.},  //
        std::array{0., 0., 0., 0., 0., j},
    };
    const auto masses_input =
        MassesInput({MassElement(model.GetNode(1), mass_matrix)}, {0., 0., 0.});
    auto masses = CreateMasses(masses_input);

    // Create springs
    const auto k = 10.;
    const auto springs_input = SpringsInput({SpringElement(
        std::array{model.GetNode(0), model.GetNode(1)},
        k,  // Spring stiffness coeff.
        0.  // Undeformed length
    )});
    auto springs = CreateSprings(springs_input);

    // Create elements
    auto elements = Elements{beams, masses, springs};

    // Add fixed BC to the first node
    model.AddFixedBC(model.GetNode(0));

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

    // The spring-mass system should move periodically between -2 and 2 (position of the second node)
    // with simple harmonic motion where the time period is 2 * pi * sqrt(m / k) = 1.98691765316s
    const double T = 2. * M_PI / sqrt(k / m);

    constexpr auto num_steps = 1000;

    // Set up step parameters
    constexpr bool is_dynamic_solve(true);
    constexpr size_t max_iter(6);
    constexpr double rho_inf(0.);                                // No damping
    const double step_size(T / static_cast<double>(num_steps));  // Calculate step size
    auto parameters = StepParameters(is_dynamic_solve, max_iter, step_size, rho_inf);

    auto q = Kokkos::create_mirror(state.q);

    // Run simulation for T seconds
    for (auto time_step = 1U; time_step <= 1000; ++time_step) {
        auto converged = Step(parameters, solver, elements, state, constraints);
        EXPECT_TRUE(converged);
        // Kokkos::deep_copy(q, state.q);
        // std::cout << time_step * step_size << " ," << q(1, 0) << '\n';
        if (time_step == num_steps / 2) {  // T/2
            Kokkos::deep_copy(q, state.q);
            EXPECT_EQ(q(0, 0), 0.);  // First node is fixed
            EXPECT_NEAR(
                q(1, 0), -3.9999200547674469, 1.e-12
            );  // Second node should have displacement close to -4.0 after T/2
        } else if (time_step == num_steps) {  // T
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
