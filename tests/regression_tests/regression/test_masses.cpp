#include <gtest/gtest.h>

#include "elements/elements.hpp"
#include "elements/masses/create_masses.hpp"
#include "model/model.hpp"
#include "state/set_node_external_loads.hpp"
#include "state/state.hpp"
#include "step/step.hpp"
#include "step/update_system_variables.hpp"
#include "test_utilities.hpp"
#include "types.hpp"

namespace openturbine::tests {

inline auto SetUpMasses() {
    // Create model object
    auto model = Model();

    // Set gravity
    model.SetGravity(0., 0., 9.81);

    // Add node
    const auto node_id = model.AddNode().SetPosition(0, 0, 0, 1, 0, 0, 0).Build();

    // Add mass element
    model.AddMassElement(
        node_id, {{
                     {1., 0., 0., 0., 0., 0.},
                     {0., 1., 0., 0., 0., 0.},
                     {0., 0., 1., 0., 0., 0.},
                     {0., 0., 0., 1., 0., 0.},
                     {0., 0., 0., 0., 1., 0.},
                     {0., 0., 0., 0., 0., 1.},
                 }}
    );

    // Initialize masses
    auto masses = model.CreateMasses();

    // Create state
    auto state = model.CreateState();

    auto parameters = StepParameters(false, 0, 0., 0.);
    UpdateSystemVariablesMasses(parameters, masses, state);

    return masses;
}

TEST(MassesTest, NodeInitialPosition) {
    const auto masses = SetUpMasses();
    expect_kokkos_view_2D_equal(
        masses.node_x0,
        {
            {0., 0., 0., 1., 0., 0., 0.},
        }
    );
}

TEST(MassesTest, MassMatrixInMaterialFrame) {
    const auto masses = SetUpMasses();
    expect_kokkos_view_2D_equal(
        Kokkos::subview(masses.qp_Mstar, 0, Kokkos::ALL, Kokkos::ALL),
        {
            {1., 0., 0., 0., 0., 0.},
            {0., 1., 0., 0., 0., 0.},
            {0., 0., 1., 0., 0., 0.},
            {0., 0., 0., 1., 0., 0.},
            {0., 0., 0., 0., 1., 0.},
            {0., 0., 0., 0., 0., 1.},
        }
    );
}

TEST(MassesTest, GravityVector) {
    const auto masses = SetUpMasses();
    expect_kokkos_view_1D_equal(masses.gravity, {0., 0., 9.81});
}

TEST(MassesTest, ExternalForce) {
    // Create model object
    auto model = Model();

    // Set gravity
    model.SetGravity(0., 0., 0.);

    // Add node
    const auto node_id = model.AddNode().SetPosition(0, 0, 0, 1, 0, 0, 0).Build();

    // Add mass element
    const auto m = 1.;
    const auto j = 1.;
    model.AddMassElement(
        node_id, {{
                     {m, 0., 0., 0., 0., 0.},
                     {0., m, 0., 0., 0., 0.},
                     {0., 0., m, 0., 0., 0.},
                     {0., 0., 0., j, 0., 0.},
                     {0., 0., 0., 0., j, 0.},
                     {0., 0., 0., 0., 0., j},
                 }}
    );

    // Initialize masses
    auto masses = model.CreateMasses();

    // Create state
    auto state = model.CreateState();

    // Create solution parameters
    const auto time_step = 0.001;
    auto parameters = StepParameters(true, 5, time_step, 0.0);

    auto constraints = model.CreateConstraints();
    auto elements = model.CreateElements();

    auto solver = CreateSolver(state, elements, constraints);

    const auto force_x = 5.;
    const auto torque_y = 3.;
    SetNodeExternalLoads(state, node_id, {force_x, 0., 0., 0., torque_y, 0.});

    // Run simulation for 1000 steps
    const auto n_steps = 1000;
    for (size_t i = 0; i < n_steps; ++i) {
        auto converged = Step(parameters, solver, elements, state, constraints);
        EXPECT_TRUE(converged);
    }

    // Get velocity
    auto v_host = Kokkos::create_mirror(state.v);
    Kokkos::deep_copy(v_host, state.v);

    EXPECT_NEAR(v_host(0, 0), 4.9975, 1.e-12);
    EXPECT_NEAR(v_host(0, 1), 0., 1.e-12);
    EXPECT_NEAR(v_host(0, 2), 0., 1.e-12);
    EXPECT_NEAR(v_host(0, 3), 0., 1.e-12);
    EXPECT_NEAR(v_host(0, 4), 2.9985, 1.e-12);
    EXPECT_NEAR(v_host(0, 5), 0., 1.e-12);
}

}  // namespace openturbine::tests
