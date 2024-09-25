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
#include "src/utilities/aerodynamics/aerodyn_inflow.hpp"
#include "src/utilities/controllers/discon.hpp"
#include "src/utilities/controllers/turbine_controller.hpp"
#include "src/vendor/dylib/dylib.hpp"
#include "tests/unit_tests/regression/iea15_rotor_data.hpp"
#include "tests/unit_tests/regression/test_utilities.hpp"

#ifdef OpenTurbine_ENABLE_VTK
#include "tests/unit_tests/regression/vtkout.hpp"
#endif

namespace openturbine::tests {

TEST(Milestone, IEA15RotorAeroController) {
    // Gravity vector
    constexpr auto gravity = std::array{-9.81, 0., 0.};

    // Properties
    constexpr size_t n_turbines{1};       // Number of turbines
    constexpr size_t n_blades{3};         // Number of blades in rotor
    constexpr double azimuth_init{0.};    // Azimuth angle (rad)
    constexpr double hub_height{150.};    // Hub height (meters)
    constexpr double hub_rad{3.97};       // Hub radius (meters)
    constexpr double gear_box_ratio{1.};  // Gear box ratio (-)
    // constexpr double rotor_speed_init{0.79063415025};    // Initial rotor rotational velocity
    // (rad/s)
    constexpr double rotor_speed_init{0.9};              // Initial rotor rotational velocity (rad/s)
    constexpr double hub_overhang{-12.097571763912535};  // hub overhang (meters)
    constexpr Array_3 shaft_axis = {1., 0., 0};          // Shaft along x-axis

    // Rotor angular velocity in rad/s
    constexpr double hub_wind_speed{12.0};

    // Controller parameters
    const std::string controller_shared_lib_path{"./ROSCO.dll"};
    const std::string controller_function_name{"DISCON"};
    const std::string controller_input_file_path{"IEA-15-240-RWT_DISCON.IN"};
    const std::string controller_simulation_name{"IEA-15-240-RWT"};

    // Aerodynamics and Inflow library
    const std::string adi_shared_lib_path{"./libaerodyn_inflow_c_binding.dylib"};

    // Solution parameters
    constexpr bool is_dynamic_solve{true};
    constexpr size_t max_iter{6};
    constexpr double step_size{0.01};  // seconds
    constexpr double rho_inf{0.0};
    constexpr double t_end{2.0 * M_PI / rotor_speed_init};  // 3 revolutions
    constexpr auto num_steps{static_cast<size_t>(t_end / step_size + 1.)};

    // Create model for adding nodes and constraints
    auto model = Model();

    //--------------------------------------------------------------------------
    // Controller Setup
    //--------------------------------------------------------------------------

    // Create controller object and load shared library
    auto controller = util::TurbineController(
        controller_shared_lib_path, controller_function_name, controller_input_file_path,
        controller_simulation_name
    );

    // Controller constant values
    controller.io.dt = step_size;               // Time step size (seconds)
    controller.io.pitch_actuator_type_req = 0;  // Pitch position actuator
    controller.io.pitch_control_type = 0;       // Collective pitch control
    controller.io.n_blades = n_turbines;        // Number of blades

    // Controller current values
    controller.io.time = 0.;                     // Current time (seconds)
    controller.io.azimuth_angle = azimuth_init;  // Initial azimuth
    controller.io.pitch_blade1_actual = 0.;      // Blade pitch (rad)
    controller.io.pitch_blade2_actual = 0.;      // Blade pitch (rad)
    controller.io.pitch_blade3_actual = 0.;      // Blade pitch (rad)
    controller.io.generator_speed_actual =
        rotor_speed_init * gear_box_ratio;                 // Generator speed (rad/s)
    controller.io.generator_torque_actual = 0;             // Generator torque
    controller.io.rotor_speed_actual = rotor_speed_init;   // Rotor speed (rad/s)
    controller.io.horizontal_wind_speed = hub_wind_speed;  // Hub wind speed (m/s)

    // Signal first call
    controller.io.status = 0;

    // Make first call to controller
    controller.CallController();

    // Actual torque applied to shaft
    double torque_actual{controller.io.generator_torque_command};

    //--------------------------------------------------------------------------
    // Blade nodes and elements
    //--------------------------------------------------------------------------

    // Node location [0, 1]
    constexpr auto num_nodes = node_xi.size();
    auto node_loc = std::array<double, num_nodes>{};
    std::transform(
        std::cbegin(node_xi), std::cend(node_xi), std::begin(node_loc),
        [&](const auto xi) {
            return (xi + 1.) / 2.;
        }
    );

    // Build vector of blade elements
    auto blade_list = std::array<size_t, n_turbines>{};
    std::iota(std::begin(blade_list), std::end(blade_list), 0);
    std::vector<BeamElement> beam_elems;
    constexpr double d_theta = 2. * M_PI / static_cast<double>(n_turbines);
    auto base_rot = RotationVectorToQuaternion({0., -M_PI / 2., 0.});
    constexpr Array_3 omega{
        rotor_speed_init * shaft_axis[0], rotor_speed_init * shaft_axis[1],
        rotor_speed_init * shaft_axis[2]};
    std::transform(
        std::cbegin(blade_list), std::cend(blade_list), std::back_inserter(beam_elems),
        [&](const size_t i) {
            // Define root rotation about x-axis
            const auto q_root =
                RotationVectorToQuaternion({d_theta * static_cast<double>(i) + azimuth_init, 0., 0.}
                );

            // Declare vector of beam nodes
            std::vector<BeamNode> beam_nodes;

            auto node_list = std::array<size_t, num_nodes>{};
            std::iota(std::begin(node_list), std::end(node_list), 0);
            std::transform(
                std::cbegin(node_list), std::cend(node_list), std::back_inserter(beam_nodes),
                [&](const size_t j) {
                    const auto rot =
                        QuaternionCompose(QuaternionCompose(q_root, base_rot), node_rotation[j]);
                    // Calculate node position and orientation for this blade
                    const auto pos = RotateVectorByQuaternion(
                        rot, {node_coords[j][0] + hub_rad, node_coords[j][1], node_coords[j][2]}
                    );
                    const auto v = CrossProduct(omega, pos);

                    // Create model node
                    return BeamNode(
                        node_loc[j],
                        *model.AddNode(
                            {pos[0] + hub_overhang, pos[1], pos[2] + hub_height, rot[0], rot[1],
                             rot[2], rot[3]},                                 // position
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
    // Rotor nodes
    //--------------------------------------------------------------------------

    auto shaft_base_node = model.AddNode({0., 0., hub_height, 1., 0., 0., 0.});
    auto azimuth_node = model.AddNode({0., 0., hub_height, 1., 0., 0., 0.});
    auto hub_node = model.AddNode({hub_overhang, 0., hub_height, 1., 0., 0., 0.});

    //--------------------------------------------------------------------------
    // Constraints
    //--------------------------------------------------------------------------

    // Fix shaft base displacement
    model.AddFixedBC(*shaft_base_node);

    // Add revolute joint between shaft base and azimuth node, rotation about shaft axis,
    // connect torque to generator torque command
    auto azimuth_constraint = model.AddRevoluteJointConstraint(
        *shaft_base_node, *azimuth_node, shaft_axis, &torque_actual
    );

    // Add rigid constraint between azimuth node and hub
    model.AddRigidJointConstraint(*azimuth_node, *hub_node);

    // Add rotation control constraints between hub and blade root nodes
    for (const auto& beam_elem : beam_elems) {
        // Calculate pitch axis from hub node to blade root node
        const Array_3 pitch_axis{
            hub_node->x[0] - beam_elem.nodes[0].node.x[0],
            hub_node->x[1] - beam_elem.nodes[0].node.x[1],
            hub_node->x[2] - beam_elem.nodes[0].node.x[2],
        };
        model.AddRotationControl(
            *hub_node, beam_elem.nodes[0].node, pitch_axis, &controller.io.pitch_collective_command
        );
    }

    // Create constraints object
    auto constraints = Constraints(model.GetConstraints());

    //--------------------------------------------------------------------------
    // AeroDyn / InflowWind library
    //--------------------------------------------------------------------------

    auto adi = util::AeroDynInflowLibrary(adi_shared_lib_path);

    adi.turbine_settings.n_turbines = n_turbines;
    adi.turbine_settings.n_blades = n_blades;

    adi.sim_controls.transpose_DCM = false;
    adi.sim_controls.debug_level = 4;
    adi.PreInitialize();

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
    const std::filesystem::path step_dir("steps/IEA15RotorAeroController");
    RemoveDirectoryWithRetries(step_dir);
    std::filesystem::create_directory(step_dir);

    // Write quadrature point global positions to file and VTK
    // Write vtk visualization file
    BeamsWriteVTK(beams, step_dir / "step_0000.vtu");
#endif

    // Initialize rotor speed
    auto rotor_speed = rotor_speed_init;
    auto azimuth = azimuth_init;

    // Perform time steps and check for convergence within max_iter iterations
    for (size_t i = 1; i < num_steps; ++i) {
        const auto generator_speed = rotor_speed * gear_box_ratio;
        const auto generator_power = generator_speed * torque_actual;

        // Set controller inputs and call controller to get commands for this step
        controller.io.status = 1;                                 // Subsequent call
        controller.io.time = step_size * static_cast<double>(i);  // Current time (seconds)
        controller.io.azimuth_angle = azimuth;                    // Current azimuth angle (rad)
        controller.io.pitch_blade1_actual = controller.io.pitch_collective_command;
        controller.io.pitch_blade2_actual = controller.io.pitch_collective_command;
        controller.io.pitch_blade3_actual = controller.io.pitch_collective_command;
        controller.io.generator_speed_actual = generator_speed;   // Generator speed (rad/s)
        controller.io.electrical_power_actual = generator_power;  // Generator power (W)
        controller.io.generator_torque_actual =
            controller.io.generator_torque_command;            // Generator torque (N-m)
        controller.io.rotor_speed_actual = rotor_speed;        // Rotor speed (rad/s)
        controller.io.horizontal_wind_speed = hub_wind_speed;  // Hub wind speed (m/s)
        controller.CallController();

        // Take step
        auto converged = Step(parameters, solver, beams, state, constraints);

        // Update rotor speed
        rotor_speed = constraints.host_output(azimuth_constraint->ID, 1);
        azimuth = constraints.host_output(azimuth_constraint->ID, 0) + azimuth_init;

        torque_actual = -1.e6;

        // Verify that step converged
        EXPECT_EQ(converged, true);

// If flag set, write quadrature point glob position to file
#ifdef OpenTurbine_ENABLE_VTK
        // Write VTK output to file
        auto tmp = std::to_string(i);
        BeamsWriteVTK(
            beams,
            step_dir / (std::string("step_") + std::string(4 - tmp.size(), '0') + tmp + ".vtu")
        );
#endif
    }
}

}  // namespace openturbine::tests
