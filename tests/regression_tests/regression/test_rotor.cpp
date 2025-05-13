#include <array>
#include <filesystem>
#include <fstream>
#include <initializer_list>
#include <iostream>

#include <gtest/gtest.h>

#include "iea15_rotor_data.hpp"
#include "model/model.hpp"
#include "solver/solver.hpp"
#include "state/state.hpp"
#include "step/step.hpp"
#include "test_utilities.hpp"
#include "types.hpp"
#include "utilities/controllers/discon.hpp"
#include "utilities/controllers/turbine_controller.hpp"
#include "vendor/dylib/dylib.hpp"

namespace openturbine::tests {

TEST(RotorTest, IEA15RotorOnly) {
    constexpr auto gravity = std::array{-9.81, 0., 0.};
    constexpr auto omega = std::array{0., 0., -0.79063415025};  // rotor angular velocity (rad/s)

    // Solution parameters
    constexpr bool is_dynamic_solve(true);
    constexpr size_t max_iter(6);
    constexpr double step_size(0.01);  // seconds
    constexpr double rho_inf(0.);
    constexpr double t_end(0.1);
    constexpr auto num_steps = static_cast<size_t>(t_end / step_size + 1.);
    constexpr auto num_nodes = 11UL;  // 11 nodes per blade

    // Node locations [0, 1]
    auto node_loc = std::array<double, num_nodes>{};
    std::transform(
        std::cbegin(node_xi), std::cend(node_xi), std::begin(node_loc),
        [](const auto xi) {
            return (xi + 1.) / 2.;
        }
    );

    // Create model for adding nodes and constraints
    auto model = Model();
    model.SetGravity(gravity[0], gravity[1], gravity[2]);

    // Create array of blade indices [0,1,2] for iterating over the 3 blades
    constexpr size_t num_blades{3};
    auto blade_list = std::array<size_t, num_blades>{};
    std::iota(std::begin(blade_list), std::end(blade_list), 0);

    // Create nodes for each blade by:
    // 1. Calculating node positions by rotating blade coordinates around hub by 120° per blade
    // 2. Setting node orientations by composing blade root rotation with local node rotation
    // 3. Setting node velocities based on prescribed rotor angular velocity (omega)
    constexpr double hub_radius{3.97};  // meters
    std::vector<size_t> beam_elem_ids;
    std::transform(
        std::cbegin(blade_list), std::cend(blade_list), std::back_inserter(beam_elem_ids),
        [&](const size_t i) {
            // Define root rotation
            const auto q_root = RotationVectorToQuaternion(
                {0., 0., -2. * M_PI * static_cast<double>(i) / static_cast<double>(num_blades)}
            );

            std::vector<size_t> beam_node_ids;

            auto node_list = std::array<size_t, num_nodes>{};
            std::iota(std::begin(node_list), std::end(node_list), 0);
            std::transform(
                std::cbegin(node_list), std::cend(node_list), std::back_inserter(beam_node_ids),
                [&](const size_t j) {
                    // Calculate node position and orientation for this blade
                    const auto pos = RotateVectorByQuaternion(
                        q_root,
                        {node_coords[j][0] + hub_radius, node_coords[j][1], node_coords[j][2]}
                    );
                    const auto rot = QuaternionCompose(q_root, node_rotation[j]);
                    const auto v = CrossProduct(omega, pos);

                    // Create model node
                    return model.AddNode()
                        .SetElemLocation(node_loc[j])
                        .SetPosition(pos[0], pos[1], pos[2], rot[0], rot[1], rot[2], rot[3])
                        .SetVelocity(v[0], v[1], v[2], omega[0], omega[1], omega[2])
                        .Build();
                }
            );

            // Add beam element
            return model.AddBeamElement(beam_node_ids, material_sections, trapz_quadrature);
        }
    );

    // Define hub node and associated constraints
    auto prescribed_bc_ids = std::vector<size_t>{};
    std::transform(
        beam_elem_ids.cbegin(), beam_elem_ids.cend(), std::back_inserter(prescribed_bc_ids),
        [&model](const auto& beam_elem_id) {
            return model.AddPrescribedBC(model.GetBeamElement(beam_elem_id).node_ids[0]);
        }
    );

    // Create solver with initial node state
    auto parameters = StepParameters(is_dynamic_solve, max_iter, step_size, rho_inf);

    // Create solver, elements, constraints, and state
    auto [state, elements, constraints, solver] = model.CreateSystemWithSolver();

#ifdef OpenTurbine_WRITE_OUTPUTS
    // Create output directory if it doesn't exist
    auto output_dir = std::string("RotorTest.IEA15RotorOnly");
    std::filesystem::create_directories(output_dir);

    // Export mesh connectivity to yaml
    model.ExportMeshConnectivityToYAML(output_dir + "/mesh_connectivity.yaml");

    // Set NetCDF output writer
    model.SetupOutputs(output_dir + "/rotor_test.nc", num_blades * num_nodes);
    model.WriteOutputsAtTimestep(state, 0);
#endif

    // Perform time steps and check for convergence within max_iter iterations
    for (size_t i = 0; i < num_steps; ++i) {
        // Calculate hub rotation for this time step
        const auto q_hub = RotationVectorToQuaternion(
            {omega[0] * step_size * static_cast<double>(i + 1),
             omega[1] * step_size * static_cast<double>(i + 1),
             omega[2] * step_size * static_cast<double>(i + 1)}
        );

        // Define hub translation/rotation displacement
        const auto u_hub = std::array{0., 0., 0., q_hub[0], q_hub[1], q_hub[2], q_hub[3]};

        // Update prescribed displacement constraint on beam root nodes
        for (const auto bc_id : prescribed_bc_ids) {
            constraints.UpdateDisplacement(bc_id, u_hub);
        }

        // Take step
        auto converged = Step(parameters, solver, elements, state, constraints);
        EXPECT_EQ(converged, true);

#ifdef OpenTurbine_WRITE_OUTPUTS
        model.WriteOutputsAtTimestep(state, i + 1);
#endif
    }
}

TEST(RotorTest, IEA15RotorHub) {
    // Gravity vector
    constexpr auto gravity = std::array{-9.81, 0., 0.};

    // Rotor angular velocity in rad/s
    constexpr auto omega = std::array{0., 0., -0.79063415025};

    // Solution parameters
    constexpr bool is_dynamic_solve(true);
    constexpr size_t max_iter(6);
    constexpr double step_size(0.01);  // seconds
    constexpr double rho_inf(0.0);
    constexpr double t_end(0.1);
    constexpr auto num_steps = static_cast<size_t>(t_end / step_size + 1.0);
    constexpr auto num_nodes = node_xi.size();
    //    constexpr auto num_nodes = 11UL;

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
    model.SetGravity(gravity[0], gravity[1], gravity[2]);

    // Build vector of nodes (straight along x axis, no rotation)
    // Calculate displacement, velocity, acceleration assuming a
    // 1 rad/s angular velocity around the z axis
    constexpr size_t num_blades = 3;
    auto blade_list = std::array<size_t, num_blades>{};
    std::iota(std::begin(blade_list), std::end(blade_list), 0);

    // Hub radius (meters)
    constexpr double hub_radius{3.97};

    std::vector<size_t> beam_elem_ids;
    std::transform(
        std::cbegin(blade_list), std::cend(blade_list), std::back_inserter(beam_elem_ids),
        [&](const size_t i) {
            // Define root rotation
            const auto q_root = RotationVectorToQuaternion(
                {0., 0., -2. * M_PI * static_cast<double>(i) / static_cast<double>(num_blades)}
            );

            // Declare vector of beam nodes
            std::vector<size_t> beam_node_ids;

            auto node_list = std::array<size_t, num_nodes>{};
            std::iota(std::begin(node_list), std::end(node_list), 0);
            std::transform(
                std::cbegin(node_list), std::cend(node_list), std::back_inserter(beam_node_ids),
                [&](const size_t j) {
                    // Calculate node position and orientation for this blade
                    const auto pos = RotateVectorByQuaternion(
                        q_root,
                        {node_coords[j][0] + hub_radius, node_coords[j][1], node_coords[j][2]}
                    );
                    const auto rot = QuaternionCompose(q_root, node_rotation[j]);
                    const auto v = CrossProduct(omega, pos);

                    // Create model node
                    return model.AddNode()
                        .SetElemLocation(node_loc[j])
                        .SetPosition(pos[0], pos[1], pos[2], rot[0], rot[1], rot[2], rot[3])
                        .SetVelocity(v[0], v[1], v[2], omega[0], omega[1], omega[2])
                        .Build();
                }
            );

            // Add beam element
            return model.AddBeamElement(beam_node_ids, material_sections, trapz_quadrature);
        }
    );

    // Define hub node and associated constraints
    auto hub_node_id = model.AddNode().SetPosition(0., 0., 0., 1., 0., 0., 0.).Build();
    for (const auto& beam_elem_id : beam_elem_ids) {
        model.AddRigidJointConstraint({hub_node_id, model.GetBeamElement(beam_elem_id).node_ids[0]});
    }
    auto hub_bc_id = model.AddPrescribedBC(hub_node_id);

    // Create solver with initial node state
    auto parameters = StepParameters(is_dynamic_solve, max_iter, step_size, rho_inf);

    // Create solver, elements, constraints, and state
    auto [state, elements, constraints, solver] = model.CreateSystemWithSolver();

#ifdef OpenTurbine_WRITE_OUTPUTS
    // Create output directory if it doesn't exist
    auto output_dir = std::string("RotorTest.IEA15RotorHub");
    std::filesystem::create_directories(output_dir);

    // Export mesh connectivity to yaml
    model.ExportMeshConnectivityToYAML(output_dir + "/mesh_connectivity.yaml");

    // Set NetCDF output writer
    auto total_num_nodes = num_blades * num_nodes + 1;  // +1 for hub node
    model.SetupOutputs(output_dir + "/rotor_test.nc", total_num_nodes);
    model.WriteOutputsAtTimestep(state, 0);
#endif

    // Perform time steps and check for convergence within max_iter iterations
    for (size_t i = 0; i < num_steps; ++i) {
        // Calculate hub rotation for this time step
        const auto q_hub = RotationVectorToQuaternion(
            {omega[0] * step_size * static_cast<double>(i + 1),
             omega[1] * step_size * static_cast<double>(i + 1),
             omega[2] * step_size * static_cast<double>(i + 1)}
        );

        // Define hub translation/rotation displacement
        const auto u_hub = std::array{0., 0., 0., q_hub[0], q_hub[1], q_hub[2], q_hub[3]};

        // Update prescribed displacement constraint on hub
        constraints.UpdateDisplacement(hub_bc_id, u_hub);

        // Take step
        auto converged = Step(parameters, solver, elements, state, constraints);

        // Verify that step converged
        EXPECT_EQ(converged, true);

#ifdef OpenTurbine_WRITE_OUTPUTS
        model.WriteOutputsAtTimestep(state, i + 1);
#endif
    }
}

TEST(RotorTest, IEA15RotorController) {
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
    model.SetGravity(gravity[0], gravity[1], gravity[2]);

    // Build vector of nodes (straight along x axis, no rotation)
    // Calculate displacement, velocity, acceleration assuming a
    // 1 rad/s angular velocity around the z axis
    constexpr size_t num_blades = 3;
    auto blade_list = std::array<size_t, num_blades>{};
    std::iota(std::begin(blade_list), std::end(blade_list), 0);

    // Hub radius (meters)
    constexpr double hub_radius{3.97};

    std::vector<size_t> beam_elem_ids;
    std::transform(
        std::cbegin(blade_list), std::cend(blade_list), std::back_inserter(beam_elem_ids),
        [&](const size_t i) {
            // Define root rotation
            const auto q_root = RotationVectorToQuaternion(
                {0., 0., -2. * M_PI * static_cast<double>(i) / static_cast<double>(num_blades)}
            );

            // Declare vector of beam nodes
            std::vector<size_t> beam_node_ids;

            auto node_list = std::array<size_t, num_nodes>{};
            std::iota(std::begin(node_list), std::end(node_list), 0);
            std::transform(
                std::cbegin(node_list), std::cend(node_list), std::back_inserter(beam_node_ids),
                [&](const size_t j) {
                    // Calculate node position and orientation for this blade
                    const auto pos = RotateVectorByQuaternion(
                        q_root,
                        {node_coords[j][0] + hub_radius, node_coords[j][1], node_coords[j][2]}
                    );
                    const auto rot = QuaternionCompose(q_root, node_rotation[j]);
                    const auto v = CrossProduct(omega, pos);

                    // Create model node
                    return model.AddNode()
                        .SetElemLocation(node_loc[j])
                        .SetPosition(pos[0], pos[1], pos[2], rot[0], rot[1], rot[2], rot[3])
                        .SetVelocity(v[0], v[1], v[2], omega[0], omega[1], omega[2])
                        .Build();
                }
            );

            // Add beam element
            return model.AddBeamElement(beam_node_ids, material_sections, trapz_quadrature);
        }
    );

    // Add logic related to TurbineController
    // provide shared library path and controller function name to clamp
    const auto shared_lib_path = std::string{"./DISCON_ROTOR_TEST_CONTROLLER.dll"};
    const auto controller_function_name = std::string{"PITCH_CONTROLLER"};

    // create an instance of TurbineController
    auto controller = util::TurbineController(
        shared_lib_path, controller_function_name, "test_input_file", "test_output_file"
    );

    // Pitch control variable
    auto blade_pitch_command = std::array<double*, 3>{
        &controller.io.pitch_blade1_command, &controller.io.pitch_blade2_command,
        &controller.io.pitch_blade3_command
    };

    // Define hub node and associated constraints
    auto hub_node_id = model.AddNode().SetPosition(0., 0., 0., 1., 0., 0., 0.).Build();
    for (size_t i = 0; i < beam_elem_ids.size(); ++i) {
        const auto q_root = RotationVectorToQuaternion(
            {0., 0., -2. * M_PI * static_cast<double>(i) / static_cast<double>(num_blades)}
        );
        const auto pitch_axis = RotateVectorByQuaternion(q_root, {1., 0., 0.});
        model.AddRotationControl(
            {hub_node_id, model.GetBeamElement(beam_elem_ids[i]).node_ids[0]}, pitch_axis,
            blade_pitch_command[i]
        );
    }
    auto hub_bc_id = model.AddPrescribedBC(hub_node_id);

    // Create solver, elements, constraints, and state
    auto parameters = StepParameters(is_dynamic_solve, max_iter, step_size, rho_inf);
    auto [state, elements, constraints, solver] = model.CreateSystemWithSolver();

#ifdef OpenTurbine_WRITE_OUTPUTS
    // Create output directory if it doesn't exist
    auto output_dir = std::string("RotorTest.IEA15RotorController");
    std::filesystem::create_directories(output_dir);

    // Export mesh connectivity to yaml
    model.ExportMeshConnectivityToYAML(output_dir + "/mesh_connectivity.yaml");

    // Set NetCDF output writer
    auto total_num_nodes = num_blades * num_nodes + 1;  // +1 for hub node
    model.SetupOutputs(output_dir + "/rotor_test.nc", total_num_nodes);
    model.WriteOutputsAtTimestep(state, 0);
#endif

    // Perform time steps and check for convergence within max_iter iterations
    for (size_t i = 0; i < num_steps; ++i) {
        // Time at end of step
        const double t = step_size * static_cast<double>(i + 1);

        // Calculate hub rotation for this time step
        const auto q_hub = RotationVectorToQuaternion({omega[0] * t, omega[1] * t, omega[2] * t});

        // Update prescribed displacement constraint on hub
        const auto u_hub = std::array{0., 0., 0., q_hub[0], q_hub[1], q_hub[2], q_hub[3]};
        constraints.UpdateDisplacement(hub_bc_id, u_hub);

        // Update time in controller
        controller.io.time = t;

        // call controller to get signals for this step
        controller.CallController();

        // Take step
        auto converged = Step(parameters, solver, elements, state, constraints);

        // Verify that step converged
        EXPECT_EQ(converged, true);

#ifdef OpenTurbine_WRITE_OUTPUTS
        model.WriteOutputsAtTimestep(state, i + 1);
#endif
    }
}

}  // namespace openturbine::tests
