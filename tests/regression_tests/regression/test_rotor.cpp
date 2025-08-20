#include <array>
#include <fstream>
#include <iostream>

#include <gtest/gtest.h>

#include "iea15_rotor_data.hpp"
#include "model/model.hpp"
#include "step/step.hpp"
#include "utilities/controllers/turbine_controller.hpp"

namespace openturbine::tests {

auto ComputeIEA15NodeLocations() {
    constexpr auto num_nodes = node_xi.size();
    auto node_loc = std::array<double, num_nodes>{};
    std::transform(
        std::cbegin(node_xi), std::cend(node_xi), std::begin(node_loc),
        [&](const auto xi) {
            return (xi + 1.) / 2.;
        }
    );
    return node_loc;
}

template <size_t num_blades>
Model CreateIEA15Blades(const std::array<double, 3>& omega) {
    auto model = Model();

    // Set gravity in model
    model.SetGravity(-9.81, 0., 0.);

    // Node location [0, 1]
    const auto node_loc = ComputeIEA15NodeLocations();

    constexpr auto num_nodes = node_loc.size();

    // Hub radius (meters)
    constexpr auto hub_rad = std::array{3.97, 0., 0.};
    const auto velocity = std::array{0., 0., 0., omega[0], omega[1], omega[2]};
    constexpr auto origin = std::array{0., 0., 0.};

    for (auto blade_number = 0U; blade_number < num_blades; ++blade_number) {
        const auto rotation_quaternion = math::RotationVectorToQuaternion(
            {0., 0., -2. * M_PI * static_cast<double>(blade_number) / static_cast<double>(num_blades)
            }
        );

        auto beam_node_ids = std::vector<size_t>(num_nodes);

        auto node_list = std::array<size_t, num_nodes>{};
        std::iota(std::begin(node_list), std::end(node_list), 0);
        std::transform(
            std::cbegin(node_list), std::cend(node_list), std::begin(beam_node_ids),
            [&](const size_t j) {
                return model.AddNode()
                    .SetElemLocation(node_loc[j])
                    .SetPosition({node_coords[j]})
                    .Build();
            }
        );
        auto blade_elem_id =
            model.AddBeamElement(beam_node_ids, material_sections, trapz_quadrature);
        model.TranslateBeam(blade_elem_id, hub_rad);
        model.RotateBeamAboutPoint(blade_elem_id, rotation_quaternion, origin);
        model.SetBeamVelocityAboutPoint(blade_elem_id, velocity, origin);
    }

    return model;
}

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
    // Rotor angular velocity in rad/s
    constexpr auto omega = std::array{0., 0., -0.79063415025};

    // Solution parameters
    constexpr bool is_dynamic_solve(true);
    constexpr size_t max_iter(6);
    constexpr double step_size(0.01);  // seconds
    constexpr double rho_inf(0.);
    constexpr double t_end(0.1);
    constexpr auto num_steps = static_cast<size_t>(t_end / step_size + 1.0);

    constexpr size_t num_blades = 3;
    auto model = CreateIEA15Blades<num_blades>(omega);

    auto prescribed_bc_ids = std::array<size_t, num_blades>{};
    std::transform(
        std::cbegin(model.GetBeamElements()), std::cend(model.GetBeamElements()),
        std::begin(prescribed_bc_ids),
        [&model](const auto& beam_elem) {
            return model.AddPrescribedBC(beam_elem.node_ids[0]);
        }
    );

    // Create solver with initial node state
    auto parameters = StepParameters(is_dynamic_solve, max_iter, step_size, rho_inf);

    // Create solver, elements, constraints, and state
    auto [state, elements, constraints, solver] = model.CreateSystemWithSolver<>();

    // Perform time steps and check for convergence within max_iter iterations
    for (auto i = 0U; i < num_steps; ++i) {
        // Calculate hub rotation for this time step
        const auto q_hub = math::RotationVectorToQuaternion(
            {omega[0] * step_size * static_cast<double>(i + 1),
             omega[1] * step_size * static_cast<double>(i + 1),
             omega[2] * step_size * static_cast<double>(i + 1)}
        );

        // Define hub translation/rotation displacement
        const auto u_hub = std::array<double, 7>{0., 0., 0., q_hub[0], q_hub[1], q_hub[2], q_hub[3]};

        // Update prescribed displacement constraint on beam root nodes
        for (const auto bc_id : prescribed_bc_ids) {
            constraints.UpdateDisplacement(bc_id, u_hub);
        }

        // Take step
        auto converged = Step(parameters, solver, elements, state, constraints);
        EXPECT_EQ(converged, true);
    }

    const auto q = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), state.q);

    // Check root nodes
    EXPECT_NEAR(q(0, 0), 0., 1.e-13);
    EXPECT_NEAR(q(0, 1), 0., 1.e-13);
    EXPECT_NEAR(q(0, 2), 0., 1.e-13);
    EXPECT_NEAR(q(0, 3), 0.99905468165654443, 1.e-13);
    EXPECT_NEAR(q(0, 4), 0., 1.e-13);
    EXPECT_NEAR(q(0, 5), 0., 1.e-13);
    EXPECT_NEAR(q(0, 6), -0.043471175048988092, 1.e-13);

    EXPECT_NEAR(q(11, 0), q(0, 0), 1.e-13);
    EXPECT_NEAR(q(11, 1), q(0, 1), 1.e-13);
    EXPECT_NEAR(q(11, 2), q(0, 2), 1.e-13);
    EXPECT_NEAR(q(11, 3), q(0, 3), 1.e-13);
    EXPECT_NEAR(q(11, 4), q(0, 4), 1.e-13);
    EXPECT_NEAR(q(11, 5), q(0, 5), 1.e-13);
    EXPECT_NEAR(q(11, 6), q(0, 6), 1.e-13);

    EXPECT_NEAR(q(22, 0), q(0, 0), 1.e-13);
    EXPECT_NEAR(q(22, 1), q(0, 1), 1.e-13);
    EXPECT_NEAR(q(22, 2), q(0, 2), 1.e-13);
    EXPECT_NEAR(q(22, 3), q(0, 3), 1.e-13);
    EXPECT_NEAR(q(22, 4), q(0, 4), 1.e-13);
    EXPECT_NEAR(q(22, 5), q(0, 5), 1.e-13);
    EXPECT_NEAR(q(22, 6), q(0, 6), 1.e-13);

    // Tip Node
    EXPECT_NEAR(q(10, 0), -0.46734512264502198, 1.e-13);
    EXPECT_NEAR(q(10, 1), -10.510711492795387, 1.e-13);
    EXPECT_NEAR(q(10, 2), -0.067517209683264995, 1.e-13);
    EXPECT_NEAR(q(10, 3), 0.99897571548945407, 1.e-13);
    EXPECT_NEAR(q(10, 4), -0.0037457241431271423, 1.e-13);
    EXPECT_NEAR(q(10, 5), 0.0040838058169996218, 1.e-13);
    EXPECT_NEAR(q(10, 6), -0.044908929435304161, 1.e-13);

    EXPECT_NEAR(q(21, 0), -8.9078971370783169, 1.e-13);
    EXPECT_NEAR(q(21, 1), 5.685448601681232, 1.e-13);
    EXPECT_NEAR(q(21, 2), -0.075564876118098229, 1.e-13);
    EXPECT_NEAR(q(21, 3), 0.99897027934209304, 1.e-13);
    EXPECT_NEAR(q(21, 4), 0.0056252825720437376, 1.e-13);
    EXPECT_NEAR(q(21, 5), 0.0014726673754152763, 1.e-13);
    EXPECT_NEAR(q(21, 6), -0.044995204610788583, 1.e-13);

    EXPECT_NEAR(q(32, 0), 9.2903477400425523, 1.e-13);
    EXPECT_NEAR(q(32, 1), 4.8313504492042556, 1.e-13);
    EXPECT_NEAR(q(32, 2), -0.074844967147025251, 1.e-13);
    EXPECT_NEAR(q(32, 3), 0.99897383015202312, 1.e-13);
    EXPECT_NEAR(q(32, 4), -0.0015569045486905323, 1.e-13);
    EXPECT_NEAR(q(32, 5), -0.005622374121860842, 1.e-13);
    EXPECT_NEAR(q(32, 6), -0.044913824473725598, 1.e-13);
}

TEST(RotorTest, IEA15RotorHub) {
    // Rotor angular velocity in rad/s
    constexpr auto omega = std::array{0., 0., -0.79063415025};

    // Solution parameters
    constexpr bool is_dynamic_solve(true);
    constexpr size_t max_iter(6);
    constexpr double step_size(0.01);  // seconds
    constexpr double rho_inf(0.0);
    constexpr double t_end(0.1);
    constexpr auto num_steps = static_cast<size_t>(t_end / step_size + 1.0);

    // Build vector of nodes (straight along x axis, no rotation)
    // Calculate displacement, velocity, acceleration assuming a
    // 1 rad/s angular velocity around the z axis
    constexpr size_t num_blades = 3;
    auto model = CreateIEA15Blades<num_blades>(omega);

    // Define hub node and associated constraints
    auto hub_node_id = model.AddNode().SetPosition(0., 0., 0., 1., 0., 0., 0.).Build();
    for (const auto& beam_elem : model.GetBeamElements()) {
        model.AddRigidJointConstraint({hub_node_id, beam_elem.node_ids[0]});
    }
    auto hub_bc_id = model.AddPrescribedBC(hub_node_id);

    // Create solver with initial node state
    auto parameters = StepParameters(is_dynamic_solve, max_iter, step_size, rho_inf);

    // Create solver, elements, constraints, and state
    auto [state, elements, constraints, solver] = model.CreateSystemWithSolver<>();

    // Perform time steps and check for convergence within max_iter iterations
    for (auto i = 0U; i < num_steps; ++i) {
        // Calculate hub rotation for this time step
        const auto q_hub = math::RotationVectorToQuaternion(
            {omega[0] * step_size * static_cast<double>(i + 1),
             omega[1] * step_size * static_cast<double>(i + 1),
             omega[2] * step_size * static_cast<double>(i + 1)}
        );

        // Define hub translation/rotation displacement
        const auto u_hub = std::array<double, 7>{0., 0., 0., q_hub[0], q_hub[1], q_hub[2], q_hub[3]};

        // Update prescribed displacement constraint on hub
        constraints.UpdateDisplacement(hub_bc_id, u_hub);

        // Take step
        auto converged = Step(parameters, solver, elements, state, constraints);

        // Verify that step converged
        EXPECT_EQ(converged, true);
    }
}

TEST(RotorTest, IEA15RotorController) {
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

    constexpr size_t num_blades = 3;
    auto model = CreateIEA15Blades<num_blades>(omega);

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
    for (const auto& beam_elem : model.GetBeamElements()) {
        const auto rotation_fraction =
            static_cast<double>(beam_elem.ID) / static_cast<double>(num_blades);
        const auto q_root =
            math::RotationVectorToQuaternion({0., 0., -2. * M_PI * rotation_fraction});
        const auto pitch_axis = math::RotateVectorByQuaternion(q_root, {1., 0., 0.});
        model.AddRotationControl(
            {hub_node_id, beam_elem.node_ids[0]}, pitch_axis, blade_pitch_command[beam_elem.ID]
        );
    }
    auto hub_bc_id = model.AddPrescribedBC(hub_node_id);

    // Create solver, elements, constraints, and state
    auto parameters = StepParameters(is_dynamic_solve, max_iter, step_size, rho_inf);
    auto [state, elements, constraints, solver] = model.CreateSystemWithSolver<>();

    // Perform time steps and check for convergence within max_iter iterations
    for (auto i = 0U; i < num_steps; ++i) {
        // Time at end of step
        const double t = step_size * static_cast<double>(i + 1);

        // Calculate hub rotation for this time step
        const auto q_hub =
            math::RotationVectorToQuaternion({omega[0] * t, omega[1] * t, omega[2] * t});

        // Update prescribed displacement constraint on hub
        const auto u_hub = std::array<double, 7>{0., 0., 0., q_hub[0], q_hub[1], q_hub[2], q_hub[3]};
        constraints.UpdateDisplacement(hub_bc_id, u_hub);

        // Update time in controller
        controller.io.time = t;

        // call controller to get signals for this step
        controller.CallController();

        // Take step
        auto converged = Step(parameters, solver, elements, state, constraints);

        // Verify that step converged
        EXPECT_EQ(converged, true);
    }
}

TEST(RotorTest, IEA15RotorHost) {
    // Rotor angular velocity in rad/s
    constexpr auto omega = std::array{0., 0., -0.79063415025};

    // Solution parameters
    constexpr bool is_dynamic_solve(true);
    constexpr size_t max_iter(6);
    constexpr double step_size(0.01);  // seconds
    constexpr double rho_inf(0.0);
    constexpr double t_end(0.1);
    constexpr auto num_steps = static_cast<size_t>(t_end / step_size + 1.0);

    constexpr size_t num_blades = 3;
    auto model = CreateIEA15Blades<num_blades>(omega);

    //    auto prescribed_bc_ids = std::array<size_t, num_blades>{};
    auto prescribed_bc_ids = std::vector<size_t>(num_blades);
    std::transform(
        std::cbegin(model.GetBeamElements()), std::cend(model.GetBeamElements()),
        std::begin(prescribed_bc_ids),
        [&model](const auto& beam_elem) {
            return model.AddPrescribedBC(beam_elem.node_ids[0]);
        }
    );

    // Create solver with initial node state
    auto parameters = StepParameters(is_dynamic_solve, max_iter, step_size, rho_inf);

    // Create solver, elements, constraints, and state
    using Device = Kokkos::Device<
        Kokkos::DefaultHostExecutionSpace, Kokkos::DefaultHostExecutionSpace::memory_space>;
    auto [state, elements, constraints, solver] = model.CreateSystemWithSolver<Device>();

    // Perform time steps and check for convergence within max_iter iterations
    for (auto i = 0U; i < num_steps; ++i) {
        // Calculate hub rotation for this time step
        const auto q_hub = math::RotationVectorToQuaternion(
            {omega[0] * step_size * static_cast<double>(i + 1),
             omega[1] * step_size * static_cast<double>(i + 1),
             omega[2] * step_size * static_cast<double>(i + 1)}
        );

        // Define hub translation/rotation displacement
        const auto u_hub = std::array<double, 7>{0., 0., 0., q_hub[0], q_hub[1], q_hub[2], q_hub[3]};

        // Update prescribed displacement constraint on beam root nodes
        for (const auto bc_id : prescribed_bc_ids) {
            constraints.UpdateDisplacement(bc_id, u_hub);
        }

        // Take step
        auto converged = Step(parameters, solver, elements, state, constraints);

        // Verify that step converged
        EXPECT_EQ(converged, true);
    }
}
}  // namespace openturbine::tests
