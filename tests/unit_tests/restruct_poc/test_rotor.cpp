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
#include "src/restruct_poc/solver/step.hpp"
#include "src/restruct_poc/types.hpp"
#include "src/utilities/controllers/discon.hpp"
#include "src/utilities/controllers/turbine_controller.hpp"
#include "src/vendor/dylib/dylib.hpp"

#ifdef OTURB_ENABLE_VTK
#include "vtkout.hpp"
#endif

namespace openturbine::restruct_poc::tests {

template <typename T>
void WriteMatrixToFile(const std::vector<std::vector<T>>& data, const std::string& filename) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Unable to open file: " << filename << std::endl;
        return;
    }
    file << std::setprecision(16);
    for (const auto& innerVector : data) {
        for (const auto& element : innerVector) {
            file << element << ",";
        }
        file << std::endl;
    }
    file.close();
}

TEST(RotorTest, IEA15Rotor) {
    // Flag to write output
    const bool write_output(false);

    // Gravity vector
    std::array<double, 3> gravity = {-9.81, 0., 0.};

    // Rotor angular velocity in rad/s
    const auto omega = std::array<double, 3>{0., 0., -0.79063415025};

    // Solution parameters
    const bool is_dynamic_solve(true);
    const size_t max_iter(6);
    const double step_size(0.01);  // seconds
    const double rho_inf(0.0);
    const double t_end(0.1);
    const size_t num_steps(t_end / step_size + 1.0);

    // Node location [0, 1]
    std::vector<double> node_loc;
    for (const auto& xi : node_xi) {
        node_loc.push_back((xi + 1.) / 2.);
    }

    // Create model for adding nodes and constraints
    auto model = Model_2();

    // Build vector of nodes (straight along x axis, no rotation)
    // Calculate displacement, velocity, acceleration assuming a
    // 1 rad/s angular velocity around the z axis
    const size_t num_blades = 3;
    std::vector<BeamElement> beam_elems;

    // Hub radius (meters)
    const double hub_rad{3.97};

    // Loop through blades
    for (size_t i = 0; i < num_blades; ++i) {
        // Define root rotation
        const auto q_root = RotationVectorToQuaternion({0., 0., -2. * M_PI * i / num_blades});

        // Declare vector of beam nodes
        std::vector<BeamNode> beam_nodes;

        // Loop through node locations
        for (size_t j = 0; j < node_loc.size(); ++j) {
            // Calculate node position and orientation for this blade
            const auto pos = RotateVectorByQuaternion(
                q_root, {node_coords[j][0] + hub_rad, node_coords[j][1], node_coords[j][2]}
            );
            const auto rot = QuaternionCompose(q_root, node_rotation[j]);
            const auto v = CrossProduct(omega, pos);

            // Create model node
            auto node = model.AddNode(
                {pos[0], pos[1], pos[2], rot[0], rot[1], rot[2], rot[3]},  // position
                {0., 0., 0., 1., 0., 0., 0.},                              // displacement
                {v[0], v[1], v[2], omega[0], omega[1], omega[2]}           // velocity
            );

            // Add beam node
            beam_nodes.push_back(BeamNode(node_loc[j], *node));
        }

        // Add beam element
        beam_elems.push_back(BeamElement(beam_nodes, material_sections, trapz_quadrature));

        // Set prescribed BC on root node
        // model.PrescribedBC(beam_nodes[0].node);
    }

    // Define beam initialization
    BeamsInput beams_input(beam_elems, gravity);

    // Initialize beams from element inputs
    auto beams = CreateBeams(beams_input);

    // Define hub node and associated constraints
    auto hub_node = model.AddNode({0., 0., 0., 1., 0., 0., 0});
    for (const auto& beam_elem : beam_elems) {
        model.AddRigidConstraint(*hub_node, beam_elem.nodes[0].node);
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

    Solver solver(
        is_dynamic_solve, max_iter, step_size, rho_inf, nodes_vector, constraints_vector, beams
    );

    // Remove output directory for writing step data
    std::filesystem::remove_all("steps");
    std::filesystem::create_directory("steps");

    // Transfer initial conditions to beam nodes and quadrature points
    UpdateState(beams, solver.state.q, solver.state.v, solver.state.vd);

    // Write quadrature point global positions to file and VTK
    std::vector<std::vector<double>> qp_x0;
    if (write_output) {
        qp_x0 = kokkos_view_2D_to_vector(beams.qp_x0);
        WriteMatrixToFile(qp_x0, "steps/step_0000.csv");

#ifdef OTURB_ENABLE_VTK
        // Write vtk visualization file
        BeamsWriteVTK(beams, "steps/step_0000.vtu");
#endif
    }

    // Perform time steps and check for convergence within max_iter iterations
    for (size_t i = 0; i < num_steps; ++i) {
        // Calculate hub rotation for this time step
        const auto q_hub = RotationVectorToQuaternion(
            {omega[0] * step_size * (i + 1), omega[1] * step_size * (i + 1),
             omega[2] * step_size * (i + 1)}
        );

        // Define hub translation/rotation displacement
        Array_7 u_hub({0, 0, 0, q_hub[0], q_hub[1], q_hub[2], q_hub[3]});

        // Update prescribed displacement constraint on hub
        solver.constraints.UpdateDisplacement(hub_bc->ID, u_hub);

        // Take step
        auto converged = Step(solver, beams);

        // Verify that step converged
        EXPECT_EQ(converged, true);

        // If flag set, write quadrature point glob position to file
        if (write_output) {
            auto tmp = std::to_string(i + 1);
            auto file_name = std::string("steps/step_") + std::string(4 - tmp.size(), '0') + tmp;
            auto qp_x = kokkos_view_2D_to_vector(beams.qp_u);
            for (size_t j = 0; j < qp_x.size(); ++j) {
                for (size_t k = 0; k < qp_x[0].size(); ++k) {
                    qp_x[j][k] += qp_x0[j][k];
                }
            }
            WriteMatrixToFile(qp_x, file_name + ".csv");

#ifdef OTURB_ENABLE_VTK
            // Write VTK output to file
            BeamsWriteVTK(beams, file_name + ".vtu");
#endif
        }
    }
}

TEST(RotorTest, IEA15RotorController) {
    // Flag to write output
    const bool write_output(true);

    // Gravity vector
    std::array<double, 3> gravity = {-9.81, 0., 0.};

    // Rotor angular velocity in rad/s
    const auto omega = std::array<double, 3>{0., 0., -0.79063415025};

    // Solution parameters
    const bool is_dynamic_solve(true);
    const size_t max_iter(6);
    const double step_size(0.01);  // seconds
    const double rho_inf(0.0);
    const double t_end(0.01 * 2.0 * M_PI / fabs(omega[2]));  // 3 revolutions
    const size_t num_steps(t_end / step_size + 1.0);

    // Hub radius (meters)
    const double hub_rad{3.97};

    // Node location [0, 1]
    std::vector<double> node_loc;
    for (const auto& xi : node_xi) {
        node_loc.push_back((xi + 1.) / 2.);
    }

    // Create model for adding nodes and constraints
    auto model = Model_2();

    // Build vector of nodes (straight along x axis, no rotation)
    // Calculate displacement, velocity, acceleration assuming a
    // 1 rad/s angular velocity around the z axis
    const size_t num_blades = 3;
    std::vector<BeamElement> beam_elems;

    // Loop through blades
    for (size_t i = 0; i < num_blades; ++i) {
        // Define root rotation
        const auto q_root = RotationVectorToQuaternion({0., 0., -2. * M_PI * i / num_blades});

        // Declare vector of beam nodes
        std::vector<BeamNode> beam_nodes;

        // Loop through node locations
        for (size_t j = 0; j < node_loc.size(); ++j) {
            // Calculate node position and orientation for this blade
            const auto pos = RotateVectorByQuaternion(
                q_root, {node_coords[j][0] + hub_rad, node_coords[j][1], node_coords[j][2]}
            );
            const auto rot = QuaternionCompose(q_root, node_rotation[j]);
            const auto v = CrossProduct(omega, pos);

            // Create model node
            auto node = model.AddNode(
                {pos[0], pos[1], pos[2], rot[0], rot[1], rot[2], rot[3]},  // position
                {0., 0., 0., 1., 0., 0., 0.},                              // displacement
                {v[0], v[1], v[2], omega[0], omega[1], omega[2]}           // velocity
            );

            // Add beam node
            beam_nodes.push_back(BeamNode(node_loc[j], *node));
        }

        // Add beam element
        beam_elems.push_back(BeamElement(beam_nodes, material_sections, trapz_quadrature));
    }

    // Define beam initialization
    BeamsInput beams_input(beam_elems, gravity);

    // Initialize beams from element inputs
    auto beams = CreateBeams(beams_input);

    // Add logic related to TurbineController
    // provide shared library path and controller function name to clamp
    std::string shared_lib_path = "./DISCON_ROTOR_TEST_CONTROLLER.dll";
    std::string controller_function_name = "PITCH_CONTROLLER";

    // create an instance of TurbineController
    util::TurbineController controller(
        shared_lib_path, controller_function_name, "test_input_file", "test_output_file"
    );

    // Pitch control variable
    std::vector<float*> blade_pitch_command{
        &controller.io->pitch_command_1, &controller.io->pitch_command_2,
        &controller.io->pitch_command_3};

    // Define hub node and associated constraints
    auto hub_node = model.AddNode({0., 0., 0., 1., 0., 0., 0});
    for (size_t i = 0; i < beam_elems.size(); ++i) {
        const auto q_root = RotationVectorToQuaternion({0., 0., -2. * M_PI * i / num_blades});
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

    Solver solver(
        is_dynamic_solve, max_iter, step_size, rho_inf, nodes_vector, constraints_vector, beams
    );

    // Remove output directory for writing step data
    std::filesystem::remove_all("steps");
    std::filesystem::create_directory("steps");

    // Transfer initial conditions to beam nodes and quadrature points
    UpdateState(beams, solver.state.q, solver.state.v, solver.state.vd);

    // Write quadrature point global positions to file and VTK
    std::vector<std::vector<double>> qp_x0;
    if (write_output) {
        qp_x0 = kokkos_view_2D_to_vector(beams.qp_x0);
        WriteMatrixToFile(qp_x0, "steps/step_0000.csv");

#ifdef OTURB_ENABLE_VTK
        // Write vtk visualization file
        BeamsWriteVTK(beams, "steps/step_0000.vtu");
#endif
    }

    // Perform time steps and check for convergence within max_iter iterations
    for (size_t i = 0; i < num_steps; ++i) {
        // Time at end of step
        const double t = step_size * (i + 1);

        // Calculate hub rotation for this time step
        const auto q_hub = RotationVectorToQuaternion({omega[0] * t, omega[1] * t, omega[2] * t});

        // Update prescribed displacement constraint on hub
        Array_7 u_hub({0, 0, 0, q_hub[0], q_hub[1], q_hub[2], q_hub[3]});
        solver.constraints.UpdateDisplacement(hub_bc->ID, u_hub);

        // Update time in controller
        controller.io->time = t;

        // call controller to get signals for this step
        controller.CallController();

        // Take step
        auto converged = Step(solver, beams);

        // Verify that step converged
        EXPECT_EQ(converged, true);

        // If flag set, write quadrature point glob position to file
        if (write_output) {
            auto tmp = std::to_string(i + 1);
            auto file_name = std::string("steps/step_") + std::string(4 - tmp.size(), '0') + tmp;

#ifdef OTURB_ENABLE_VTK
            // Write VTK output to file
            BeamsWriteVTK(beams, file_name + ".vtu");
#endif
        }
    }
}

}  // namespace openturbine::restruct_poc::tests
