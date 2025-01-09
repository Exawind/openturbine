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

    // Add node with initial position and velocity
    auto node_id = model.AddNode().SetVelocity(0., 0., 0., 0.5, 0.5, 1.0).Build();

    // Add masses element
    constexpr auto m = 1.0;
    model.AddMassElement(
        node_id, {{
                     {m, 0., 0., 0., 0., 0.},    //
                     {0., m, 0., 0., 0., 0.},    //
                     {0., 0., m, 0., 0., 0.},    //
                     {0., 0., 0., 1., 0., 0.},   //
                     {0., 0., 0., 0., 1., 0.},   //
                     {0., 0., 0., 0., 0., 0.5},  //
                 }}
    );

    // Set up step parameters
    constexpr bool is_dynamic_solve(true);
    constexpr size_t max_iter(6);
    constexpr double step_size(0.01);
    constexpr double rho_inf(1.0);
    auto parameters = StepParameters(is_dynamic_solve, max_iter, step_size, rho_inf);

    // Create solver, elements, constraints, and state
    auto state = model.CreateState();
    auto elements = model.CreateElements();
    auto constraints = model.CreateConstraints();
    auto solver = model.CreateSolver(state, elements, constraints);

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
