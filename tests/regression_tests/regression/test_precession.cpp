#include <gtest/gtest.h>

#include "dof_management/assemble_node_freedom_allocation_table.hpp"
#include "dof_management/compute_node_freedom_map_table.hpp"
#include "dof_management/create_constraint_freedom_table.hpp"
#include "dof_management/create_element_freedom_table.hpp"
#include "elements/beams/create_beams.hpp"
#include "elements/elements.hpp"
#include "elements/masses/create_masses.hpp"
#include "elements/springs/create_springs.hpp"
#include "math/quaternion_operations.hpp"
#include "model/model.hpp"
#include "solver/solver.hpp"
#include "state/state.hpp"
#include "step/step.hpp"
#include "step/update_system_variables.hpp"
#include "test_utilities.hpp"
#include "types.hpp"

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
    auto [state, elements, constraints, solver] = model.CreateSystemWithSolver();

    // Run simulation for 500 steps
    for (size_t i = 0; i < 500; ++i) {
        auto converged = Step(parameters, solver, elements, state, constraints);
        EXPECT_TRUE(converged);
    }

    auto q_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), state.q);
    EXPECT_NEAR(q_host(0, 0), 0., 1.e-12);
    EXPECT_NEAR(q_host(0, 1), 0., 1.e-12);
    EXPECT_NEAR(q_host(0, 2), 0., 1.e-12);
    EXPECT_NEAR(q_host(0, 3), -0.63053045128590757, 1.e-12);
    EXPECT_NEAR(q_host(0, 4), 0.60556039120583116, 1.e-12);
    EXPECT_NEAR(q_host(0, 5), -0.30157681970585326, 1.e-12);
    EXPECT_NEAR(q_host(0, 6), -0.38049886257377241, 1.e-12);
}

TEST(PrecessionTest, FinalRotation) {
    SetUpPrecessionTest();
}

}  // namespace openturbine::tests
