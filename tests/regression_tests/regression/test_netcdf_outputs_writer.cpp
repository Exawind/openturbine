#include <filesystem>

#include <gtest/gtest.h>

#include "model/model.hpp"
#include "solver/solver.hpp"
#include "step/step.hpp"
#include "utilities/netcdf/node_state_writer.hpp"

namespace openturbine::tests {

TEST(NetCDFOutputsWriterTest, SpringMassSystemOutputs) {
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

    // Add fixed BC and spring element
    model.AddFixedBC(fixed_node_id);
    const auto k = 10.;  // stiffness
    const auto l0 = 0.;  // undeformed length
    model.AddSpringElement(fixed_node_id, mass_node_id, k, l0);

    // Set up simulation parameters
    const double T = 2. * M_PI * sqrt(m / k);
    constexpr auto num_steps = 1000;
    constexpr bool is_dynamic_solve(true);
    constexpr size_t max_iter(6);
    constexpr double rho_inf(0.);
    const double step_size(T / static_cast<double>(num_steps));
    auto parameters = StepParameters(is_dynamic_solve, max_iter, step_size, rho_inf);

    // Create solver and system
    auto [state, elements, constraints, solver] = model.CreateSystemWithSolver();
    auto q = Kokkos::create_mirror(state.q);

    // Set up NodeStateWriter with 2 nodes
    const std::string output_file = "spring_mass_system.nc";
    std::filesystem::remove(output_file);  // Remove any existing file
    util::NodeStateWriter writer(output_file, true, 2);

    // Run simulation and write outputs
    for (auto time_step = 0U; time_step <= num_steps; ++time_step) {
        Kokkos::deep_copy(q, state.q);

        // Write only displacement data
        writer.WriteStateData(
            time_step,
            "u",                 // displacement
            {q(0, 0), q(1, 0)},  // x component
            {q(0, 1), q(1, 1)},  //
            {q(0, 2), q(1, 2)},  //
            {q(0, 3), q(1, 3)},  //
            {q(0, 4), q(1, 4)},  //
            {q(0, 5), q(1, 5)},  //
            {q(0, 6), q(1, 6)}   //
        );

        // Step the simulation forward
        if (time_step < num_steps) {
            auto converged = Step(parameters, solver, elements, state, constraints);
            EXPECT_TRUE(converged);
        }
    }

    EXPECT_TRUE(std::filesystem::exists(output_file));
    std::filesystem::remove(output_file);
}

}  // namespace openturbine::tests
