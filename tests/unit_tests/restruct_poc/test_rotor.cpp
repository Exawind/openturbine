#include <filesystem>
#include <fstream>
#include <initializer_list>
#include <iostream>

#include <gtest/gtest.h>

#include "iea15_rotor_data.hpp"
#include "test_utilities.hpp"

#include "src/restruct_poc/beams/beam_element.hpp"
#include "src/restruct_poc/beams/beam_node.hpp"
#include "src/restruct_poc/beams/beam_section.hpp"
#include "src/restruct_poc/beams/beams.hpp"
#include "src/restruct_poc/beams/beams_input.hpp"
#include "src/restruct_poc/beams/create_beams.hpp"
#include "src/restruct_poc/model/model.hpp"
#include "src/restruct_poc/solver/solver.hpp"
#include "src/restruct_poc/state/copy_nodes_to_state.hpp"
#include "src/restruct_poc/state/state.hpp"
#include "src/restruct_poc/step/step.hpp"
#include "src/restruct_poc/types.hpp"
#include "src/utilities/controllers/discon.hpp"
#include "src/utilities/controllers/turbine_controller.hpp"
#include "src/vendor/dylib/dylib.hpp"

#ifdef OTURB_ENABLE_VTK
#include "vtkout.hpp"
#endif

namespace openturbine::tests {

template <typename T>
void WriteMatrixToFile(const std::vector<std::vector<T>>& data, const std::string& filename) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Unable to open file: " << filename << "\n";
        return;
    }
    file << std::setprecision(16);
    for (const auto& innerVector : data) {
        for (const auto& element : innerVector) {
            file << element << ",";
        }
        file << "\n";
    }
    file.close();
}

TEST(RotorTest, IEA15Rotor) {
    // Flag to write output
    constexpr bool write_output(false);

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

    // Node location [0, 1]
    auto node_loc = std::array<double, num_nodes>{};
    std::transform(
        std::cbegin(node_xi), std::cend(node_xi), std::begin(node_loc),
        [](const auto xi) {
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

    // Define hub node and associated constraints
    auto prescribed_bc = std::vector<Constraint>{};
    std::transform(
        beam_elems.cbegin(), beam_elems.cend(), std::back_inserter(prescribed_bc),
        [&model](const auto& beam_elem) {
            return *model.AddPrescribedBC(beam_elem.nodes[0].node, {0., 0., 0.});
        }
    );

    // Create solver with initial node state
    auto parameters = StepParameters(is_dynamic_solve, max_iter, step_size, rho_inf);
    auto constraints = Constraints(model.GetConstraints());
    auto state = State(model.NumNodes());
    CopyNodesToState(state, model.GetNodes());
    auto solver = Solver(
        state.ID, beams.num_nodes_per_element, beams.node_state_indices, constraints.num_dofs,
        constraints.type, constraints.base_node_index, constraints.target_node_index, constraints.row_range
    );

    // Remove output directory for writing step data
    std::filesystem::remove_all("steps");
    std::filesystem::create_directory("steps");

    // Write quadrature point global positions to file and VTK
    if (write_output) {
#ifdef OTURB_ENABLE_VTK
        // Write vtk visualization file
        BeamsWriteVTK(beams, "steps/step_0000.vtu");
#endif
    }

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
        std::for_each(
            prescribed_bc.cbegin(), prescribed_bc.cend(),
            [&constraints, &u_hub](const auto& bc) {
                constraints.UpdateDisplacement(static_cast<size_t>(bc.ID), u_hub);
            }
        );

        // Take step
        auto converged = Step(parameters, solver, beams, state, constraints);

        // Verify that step converged
        EXPECT_EQ(converged, true);

        // If flag set, write quadrature point glob position to file
        if (write_output) {
#ifdef OTURB_ENABLE_VTK
            // Write VTK output to file
            BeamsWriteVTK(beams, file_name + ".vtu");
#endif
        }
    }
}

TEST(RotorTest, IEA15RotorHub) {
    // Flag to write output
    constexpr bool write_output(false);

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

    // Define hub node and associated constraints
    auto hub_node = model.AddNode({0., 0., 0., 1., 0., 0., 0.});
    for (const auto& beam_elem : beam_elems) {
        model.AddRigidConstraint(*hub_node, beam_elem.nodes[0].node);
    }
    auto hub_bc = model.AddPrescribedBC(*hub_node, {0., 0., 0.});

    // Create solver with initial node state
    auto parameters = StepParameters(is_dynamic_solve, max_iter, step_size, rho_inf);
    auto constraints = Constraints(model.GetConstraints());
    auto state = State(model.NumNodes());
    CopyNodesToState(state, model.GetNodes());
    auto solver = Solver(
        state.ID, beams.num_nodes_per_element, beams.node_state_indices, constraints.num_dofs,
        constraints.type, constraints.base_node_index, constraints.target_node_index, constraints.row_range
    );

    // Remove output directory for writing step data
    std::filesystem::remove_all("steps");
    std::filesystem::create_directory("steps");

    // Write quadrature point global positions to file and VTK
    if (write_output) {
#ifdef OTURB_ENABLE_VTK
        // Write vtk visualization file
        BeamsWriteVTK(beams, "steps/step_0000.vtu");
#endif
    }

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
        constraints.UpdateDisplacement(hub_bc->ID, u_hub);

        // Take step
        auto converged = Step(parameters, solver, beams, state, constraints);

        // Verify that step converged
        EXPECT_EQ(converged, true);

        // If flag set, write quadrature point glob position to file
        if (write_output) {
#ifdef OTURB_ENABLE_VTK
            // Write VTK output to file
            BeamsWriteVTK(beams, file_name + ".vtu");
#endif
        }
    }
}

TEST(RotorTest, IEA15RotorController) {
    // Flag to write output
    constexpr bool write_output(true);

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
        &controller.io.pitch_command_1, &controller.io.pitch_command_2,
        &controller.io.pitch_command_3};

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

    auto parameters = StepParameters(is_dynamic_solve, max_iter, step_size, rho_inf);
    auto constraints = Constraints(model.GetConstraints());
    auto state = State(model.NumNodes());
    CopyNodesToState(state, model.GetNodes());
    auto solver = Solver(
        state.ID, beams.num_nodes_per_element, beams.node_state_indices, constraints.num_dofs,
        constraints.type, constraints.base_node_index, constraints.target_node_index, constraints.row_range
    );

    // Remove output directory for writing step data
    std::filesystem::remove_all("steps");
    std::filesystem::create_directory("steps");

    // Write quadrature point global positions to file and VTK
    if (write_output) {
#ifdef OTURB_ENABLE_VTK
        // Write vtk visualization file
        BeamsWriteVTK(beams, "steps/step_0000.vtu");
#endif
    }

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
        if (write_output) {
#ifdef OTURB_ENABLE_VTK
            // Write VTK output to file
            auto tmp = std::to_string(i + 1);
            auto file_name = std::string("steps/step_") + std::string(4 - tmp.size(), '0') + tmp;
            BeamsWriteVTK(beams, file_name + ".vtu");
#endif
        }
    }
}

}  // namespace openturbine::tests
