#include <gtest/gtest.h>

#include "test_utilities.hpp"

#include "src/model/model.hpp"
#include "src/step/step.hpp"

namespace openturbine::tests {

inline auto SetUpHeavyTopTest() {
    auto model = Model();
    model.SetGravity(0., 0., -9.81);

    // Heavy top model parameters
    constexpr auto m = 15.;                                        // mass
    constexpr auto j = std::array{0.234375, 0.46875, 0.234375};    // inertia matrix
    const auto x = std::array<double, 3>{0., 1., 0.};              // initial position
    const auto omega = std::array<double, 3>{0., 150., -4.61538};  // initial angular velocity
    const auto x_dot = CrossProduct(omega, x);                     // {4.61538, 0., 0.}
    const auto omega_dot =
        std::array<double, 3>{661.3461692307691919, 0., 0.};  // From pilot rust code
    const auto x_ddot = std::array<double, 3>{
        0., -21.3017325444000001, -30.9608307692308244
    };  // From pilot rust code

    // Add node with initial position and velocity
    auto mass_node_id =
        model.AddNode()
            .SetPosition(x[0], x[1], x[2], 1., 0., 0., 0.)
            .SetVelocity(x_dot[0], x_dot[1], x_dot[2], omega[0], omega[1], omega[2])
            .SetAcceleration(
                x_ddot[0], x_ddot[1], x_ddot[2], omega_dot[0], omega_dot[1], omega_dot[2]
            )
            .Build();

    // Add masses element with m and J as mass matrix
    model.AddMassElement(
        mass_node_id, {{
                          {m, 0., 0., 0., 0., 0.},     // mass in x-direction
                          {0., m, 0., 0., 0., 0.},     // mass in y-direction
                          {0., 0., m, 0., 0., 0.},     // mass in z-direction
                          {0., 0., 0., j[0], 0., 0.},  // inertia xx
                          {0., 0., 0., 0., j[1], 0.},  // inertia yy
                          {0., 0., 0., 0., 0., j[2]},  // inertia zz
                      }}
    );

    // Add ground node at origin
    auto ground_node_id = model.AddNode().SetPosition(0., 0., 0., 1., 0., 0., 0.).Build();

    // Add constraints (6 DOF base node -> 3 DOF target node)
    model.AddRigidJoint6DOFsTo3DOFs({mass_node_id, ground_node_id});
    model.AddPrescribedBC6DOFsTo3DOFs(ground_node_id);

    // Set up step parameters
    constexpr bool is_dynamic_solve(true);
    constexpr size_t max_iter(6);
    constexpr double step_size(0.002);
    constexpr double rho_inf(0.9);
    auto parameters = StepParameters(is_dynamic_solve, max_iter, step_size, rho_inf);

    // Create solver, elements, constraints, and state
    auto state = model.CreateState();
    auto elements = model.CreateElements();
    auto constraints = model.CreateConstraints();
    auto solver = CreateSolver(state, elements, constraints);

    // Run simulation for 400 steps i.e. 0.8s
    for (size_t i = 0; i < 400; ++i) {
        auto converged = Step(parameters, solver, elements, state, constraints);
        EXPECT_TRUE(converged);
    }

    // Check final state at t=0.8s
    auto q_host = Kokkos::create_mirror(state.q);
    Kokkos::deep_copy(q_host, state.q);

    EXPECT_NEAR(q_host(0, 0), -0.4220299141898183, 1.e-12);
    EXPECT_NEAR(q_host(0, 1), -0.09451353137427536, 1.e-12);
    EXPECT_NEAR(q_host(0, 2), -0.04455341442645723, 1.e-12);
    EXPECT_NEAR(q_host(0, 3), -0.17794086498990777, 1.e-12);
    EXPECT_NEAR(q_host(0, 4), 0.21672292516262048, 1.e-12);
    EXPECT_NEAR(q_host(0, 5), -0.9597292673920982, 1.e-12);
    EXPECT_NEAR(q_host(0, 6), -0.016969254156485276, 1.e-12);
}

TEST(HeavyTopTest, FinalState) {
    SetUpHeavyTopTest();
}

}  // namespace openturbine::tests