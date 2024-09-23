#include <filesystem>
#include <fstream>
#include <initializer_list>
#include <iostream>

#include <gtest/gtest.h>

#include "src/beams/beam_element.hpp"
#include "src/beams/beam_node.hpp"
#include "src/beams/beam_section.hpp"
#include "src/beams/beams.hpp"
#include "src/beams/beams_input.hpp"
#include "src/beams/create_beams.hpp"
#include "src/model/model.hpp"
#include "src/solver/solver.hpp"
#include "src/state/state.hpp"
#include "src/step/step.hpp"
#include "src/types.hpp"
#include "src/utilities/controllers/discon.hpp"
#include "src/utilities/controllers/turbine_controller.hpp"
#include "src/vendor/dylib/dylib.hpp"
#include "tests/unit_tests/regression/iea15_rotor_data.hpp"
#include "tests/unit_tests/regression/test_utilities.hpp"

#ifdef OpenTurbine_ENABLE_VTK
#include "tests/unit_tests/regression/vtkout.hpp"
#endif

namespace openturbine::tests {

TEST(RotorTest, IEA15RotorAeroController) {
    // Gravity vector
    constexpr auto gravity = std::array{-9.81, 0., 0.};

    // Rotor angular velocity in rad/s
    constexpr auto angular_speed = 0.79063415025;
    constexpr auto omega = std::array{0., 0., -angular_speed};

    // Solution parameters
    constexpr bool is_dynamic_solve(true);
    constexpr size_t max_iter(6);
    constexpr double step_size(0.01);  // seconds
    constexpr double rho_inf(0.0);
    constexpr double t_end(0.01 * 2.0 * M_PI / angular_speed);  // 3 revolutions
    constexpr auto num_steps = static_cast<size_t>(t_end / step_size + 1.);
    constexpr auto num_nodes = node_xi.size();

    //--------------------------------------------------------------------------
    // Controller Setup
    //--------------------------------------------------------------------------

    const auto shared_lib_path = std::string{"./ROSCO.dll"};
    const auto controller_function_name = std::string{"DISCON"};

    auto controller =
        util::TurbineController(shared_lib_path, controller_function_name, "DISCON.IN", "");

    controller.io.status = 0;
    controller.io.time = 0.;
    controller.io.dt = 0.01;
    controller.io.rotor_speed_actual = 5.;

    controller.CallController();

    // Pitch control variables
    auto blade_pitch_command = std::array<double*, 3>{
        &controller.io.pitch_collective_command, &controller.io.pitch_collective_command,
        &controller.io.pitch_collective_command};

    //--------------------------------------------------------------------------
    // Rotor Elements
    //--------------------------------------------------------------------------

    // Node location [0, 1]
    auto node_loc = std::array<double, num_nodes>{};
    std::transform(
        std::cbegin(node_xi), std::cend(node_xi), std::begin(node_loc),
        [&](const auto xi) {
            return (xi + 1.) / 2.;
        }
    );

    // Create model for adding nodes and constraints
    auto model = Model();

    // Build vector of nodes (straight along x axis, no rotation)
    // Calculate displacement, velocity, acceleration assuming a
    // 1 rad/s angular velocity around the z axis
    constexpr size_t num_blades = 3;
    auto blade_list = std::array<size_t, num_blades>{};
    std::iota(std::begin(blade_list), std::end(blade_list), 0);

    // Hub radius (meters)
    constexpr double hub_rad{3.97};

    std::vector<BeamElement> beam_elems;
    std::transform(
        std::cbegin(blade_list), std::cend(blade_list), std::back_inserter(beam_elems),
        [&](const size_t i) {
            // Define root rotation
            const auto q_root = RotationVectorToQuaternion(
                {0., 0., -2. * M_PI * static_cast<double>(i) / static_cast<double>(num_blades)}
            );

            // Declare vector of beam nodes
            std::vector<BeamNode> beam_nodes;

            auto node_list = std::array<size_t, num_nodes>{};
            std::iota(std::begin(node_list), std::end(node_list), 0);
            std::transform(
                std::cbegin(node_list), std::cend(node_list), std::back_inserter(beam_nodes),
                [&](const size_t j) {
                    // Calculate node position and orientation for this blade
                    const auto pos = RotateVectorByQuaternion(
                        q_root, {node_coords[j][0] + hub_rad, node_coords[j][1], node_coords[j][2]}
                    );
                    const auto rot = QuaternionCompose(q_root, node_rotation[j]);
                    const auto v = CrossProduct(omega, pos);

                    // Create model node
                    return BeamNode(
                        node_loc[j],
                        *model.AddNode(
                            {pos[0], pos[1], pos[2], rot[0], rot[1], rot[2], rot[3]},  // position
                            {0., 0., 0., 1., 0., 0., 0.},                     // displacement
                            {v[0], v[1], v[2], omega[0], omega[1], omega[2]}  // velocity
                        )
                    );
                }
            );

            // Add beam element
            return BeamElement(beam_nodes, material_sections, trapz_quadrature);
        }
    );

    // Define beam initialization
    const auto beams_input = BeamsInput(beam_elems, gravity);

    // Initialize beams from element inputs
    auto beams = CreateBeams(beams_input);

    //--------------------------------------------------------------------------
    // Constraints
    //--------------------------------------------------------------------------

    // Define hub node and associated constraints
    auto hub_node = model.AddNode({0., 0., 0., 1., 0., 0., 0.});
    for (size_t i = 0; i < beam_elems.size(); ++i) {
        const auto q_root = RotationVectorToQuaternion(
            {0., 0., -2. * M_PI * static_cast<double>(i) / static_cast<double>(num_blades)}
        );
        const auto pitch_axis = RotateVectorByQuaternion(q_root, {1., 0., 0.});
        model.AddRotationControl(
            *hub_node, beam_elems[i].nodes[0].node, pitch_axis, blade_pitch_command[i]
        );
    }
    auto hub_bc = model.AddPrescribedBC(*hub_node, {0., 0., 0.});

    // Create solver with initial node state
    auto nodes_vector = std::vector<Node>{};
    for (const auto& node : model.GetNodes()) {
        nodes_vector.push_back(*node);
    }

    auto constraints_vector = std::vector<Constraint>{};
    for (const auto& constraint : model.GetConstraints()) {
        constraints_vector.push_back(*constraint);
    }

    auto constraints = Constraints(model.GetConstraints());

    //--------------------------------------------------------------------------
    // Solver
    //--------------------------------------------------------------------------

    auto state = model.CreateState();
    auto parameters = StepParameters(is_dynamic_solve, max_iter, step_size, rho_inf);
    auto solver = Solver(
        state.ID, beams.num_nodes_per_element, beams.node_state_indices, constraints.num_dofs,
        constraints.type, constraints.base_node_index, constraints.target_node_index,
        constraints.row_range
    );

    // Remove output directory for writing step data
#ifdef OpenTurbine_ENABLE_VTK
    std::filesystem::path step_dir("steps/IEA15RotorAeroController");
    RemoveDirectoryWithRetries(step_dir);
    std::filesystem::create_directory(step_dir);

    // Write quadrature point global positions to file and VTK
    // Write vtk visualization file
    BeamsWriteVTK(beams, step_dir / "step_0000.vtu");
#endif

    // Perform time steps and check for convergence within max_iter iterations
    for (size_t i = 0; i < num_steps; ++i) {
        // Time at end of step
        const double t = step_size * static_cast<double>(i + 1);

        // Calculate hub rotation for this time step
        const auto q_hub = RotationVectorToQuaternion({omega[0] * t, omega[1] * t, omega[2] * t});

        // Update prescribed displacement constraint on hub
        const auto u_hub = std::array{0., 0., 0., q_hub[0], q_hub[1], q_hub[2], q_hub[3]};
        constraints.UpdateDisplacement(hub_bc->ID, u_hub);

        // Update time in controller
        controller.io.time = t;

        // call controller to get signals for this step
        controller.CallController();

        // Take step
        auto converged = Step(parameters, solver, beams, state, constraints);

        // Verify that step converged
        EXPECT_EQ(converged, true);

// If flag set, write quadrature point glob position to file
#ifdef OpenTurbine_ENABLE_VTK
        // Write VTK output to file
        auto tmp = std::to_string(i + 1);
        BeamsWriteVTK(
            beams,
            step_dir / (std::string("step_") + std::string(4 - tmp.size(), '0') + tmp + ".vtu")
        );
#endif
    }
}

}  // namespace openturbine::tests
