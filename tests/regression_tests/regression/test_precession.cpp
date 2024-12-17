#include <gtest/gtest.h>

#include "test_utilities.hpp"

#include "src/dof_management/assemble_node_freedom_allocation_table.hpp"
#include "src/dof_management/compute_node_freedom_map_table.hpp"
#include "src/dof_management/create_constraint_freedom_table.hpp"
#include "src/dof_management/create_element_freedom_table.hpp"
#include "src/elements/beams/create_beams.hpp"
#include "src/elements/elements.hpp"
#include "src/elements/masses/create_masses.hpp"
#include "src/math/quaternion_operations.hpp"
#include "src/model/model.hpp"
#include "src/solver/solver.hpp"
#include "src/state/state.hpp"
#include "src/step/step.hpp"
#include "src/step/update_system_variables.hpp"
#include "src/types.hpp"

namespace openturbine::tests {

inline auto SetUpPrecessionTest() {
    auto model = Model();

    // Set up mass matrix (6x6 diagonal matrix)
    constexpr auto m = 1.0;
    constexpr auto mass_matrix = std::array{
        std::array{m, 0., 0., 0., 0., 0.},  std::array{0., m, 0., 0., 0., 0.},
        std::array{0., 0., m, 0., 0., 0.},  std::array{0., 0., 0., 1., 0., 0.},
        std::array{0., 0., 0., 0., 1., 0.}, std::array{0., 0., 0., 0., 0., 0.5},
    };

    // Add node with initial position and velocity
    model.AddNode(
        {0., 0., 0., 1., 0., 0., 0.},  // Initial position and orientation
        {0., 0., 0., 1., 0., 0., 0.},  // Initial displacement
        {0., 0., 0., 0.5, 0.5, 1.0},   // Initial velocity
        {0., 0., 0., 0., 0., 0.}       // Initial acceleration
    );

    // Create masses element
    const auto masses_input = MassesInput(
        {
            MassElement(model.GetNode(0), mass_matrix),
        },
        {0., 0., 0.}  // No gravity for this test
    );

    auto masses = CreateMasses(masses_input);

    const auto beams_input = BeamsInput({}, {0., 0., 0.});
    auto beams = CreateBeams(beams_input);

    // Set up step parameters
    constexpr bool is_dynamic_solve(true);
    constexpr size_t max_iter(6);
    constexpr double step_size(0.01);
    constexpr double rho_inf(1.0);
    auto parameters = StepParameters(is_dynamic_solve, max_iter, step_size, rho_inf);

    // Create solver with initial node state
    auto state = model.CreateState();
    auto constraints = Constraints(model.GetConstraints());
    auto elements = Elements{beams, masses};

    assemble_node_freedom_allocation_table(state, elements, constraints);
    compute_node_freedom_map_table(state);
    create_element_freedom_table(elements, state);
    create_constraint_freedom_table(constraints, state);

    [[maybe_unused]] auto solver = Solver(
        state.ID, state.node_freedom_allocation_table, state.node_freedom_map_table,
        elements.NumberOfNodesPerElement(), elements.NodeStateIndices(), constraints.num_dofs,
        constraints.type, constraints.base_node_freedom_table, constraints.target_node_freedom_table,
        constraints.row_range
    );

    // Run simulation for 500 steps
    for (size_t i = 0; i < 500; ++i) {
        auto converged = Step(parameters, solver, elements, state, constraints);
        EXPECT_TRUE(converged);
    }

    auto q_host = Kokkos::create_mirror(state.q);
    Kokkos::deep_copy(q_host, state.q);
    EXPECT_NEAR(q_host(0, 0), 0., 1.e-12);
    EXPECT_NEAR(q_host(0, 1), 0., 1.e-12);
    EXPECT_NEAR(q_host(0, 2), 0., 1.e-12);
    EXPECT_NEAR(q_host(0, 3), -0.6305304765029902, 1.e-12);
    EXPECT_NEAR(q_host(0, 4), 0.6055602536398981, 1.e-12);
    EXPECT_NEAR(q_host(0, 5), -0.30157705376951366, 1.e-12);
    EXPECT_NEAR(q_host(0, 6), -0.3804988542061519, 1.e-12);
}

TEST(PrecessionTest, FinalRotation) {
    SetUpPrecessionTest();
}

}  // namespace openturbine::tests
