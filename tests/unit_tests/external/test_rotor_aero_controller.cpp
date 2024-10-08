#include <algorithm>
#include <array>
#include <filesystem>
#include <fstream>
#include <initializer_list>
#include <iostream>
#include <iterator>

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
#include "tests/unit_tests/regression/vtkout.hpp"

namespace openturbine::tests {

TEST(Milestone, IEA15RotorAeroController) {
    // Gravity vector
    constexpr auto gravity = std::array{-9.81, 0., 0.};

    // Properties
    constexpr size_t n_blades{3};         // Number of blades in rotor
    constexpr double azimuth_init{0.};    // Azimuth angle (rad)
    constexpr double hub_height{150.};    // Hub height (meters)
    constexpr double hub_radius{3.97};    // Hub radius (meters)
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
    const std::string adi_shared_lib_path{"./aerodyn_inflow_c_binding.dll"};

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
    controller.io.n_blades = n_blades;          // Number of blades

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

    auto base_rot = RotationVectorToQuaternion({0., -M_PI / 2., 0.});

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
    auto blade_list = std::array<size_t, n_blades>{};
    std::iota(std::begin(blade_list), std::end(blade_list), 0);
    std::vector<BeamElement> beam_elems;
    constexpr double d_theta = 2. * M_PI / static_cast<double>(n_blades);

    constexpr Array_3 omega{
        rotor_speed_init * shaft_axis[0], rotor_speed_init * shaft_axis[1],
        rotor_speed_init * shaft_axis[2]};
    std::transform(
        std::cbegin(blade_list), std::cend(blade_list), std::back_inserter(beam_elems),
        [&](const size_t i) {
            // Define root rotation about x-axis
            const auto q_root = QuaternionCompose(
                RotationVectorToQuaternion({d_theta * static_cast<double>(i) + azimuth_init, 0., 0.}
                ),
                base_rot
            );

            // Declare vector of beam nodes
            std::vector<BeamNode> beam_nodes;

            for (size_t j = 0; j < num_nodes; ++j) {
                const auto rot = QuaternionCompose(q_root, node_rotation[j]);
                // Calculate node position and orientation for this blade
                const auto pos = RotateVectorByQuaternion(
                    q_root, {node_coords[j][0] + hub_radius, node_coords[j][1], node_coords[j][2]}
                );
                const auto v = CrossProduct(omega, pos);

                // Create model node
                beam_nodes.emplace_back(BeamNode(
                    node_loc[j], *model.AddNode(
                                     {pos[0] + hub_overhang, pos[1], pos[2] + hub_height, rot[0],
                                      rot[1], rot[2], rot[3]},      // position
                                     {0., 0., 0., 1., 0., 0., 0.},  // displacement
                                     {v[0], v[1], v[2], omega[0], omega[1], omega[2]}  // velocity
                                 )
                ));
            }

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

    // Create lambda for building blade configuration
    auto build_blade_config = [&](size_t blade_num) {
        std::vector<Array_7> node_positions;
        std::transform(
            beam_elems[blade_num].nodes.begin(), beam_elems[blade_num].nodes.end(),
            std::back_inserter(node_positions),
            [](const BeamNode& n) {
                return n.node.x;
            }
        );
        return util::TurbineConfig::BladeInitialState{
            node_positions[0],  // root node
            node_positions,     // all nodes
        };
    };

    // Define turbine initial position
    std::vector<util::TurbineConfig> turbine_configs{
        util::TurbineConfig(
            true,          // is horizontal axis wind turbine
            {0., 0., 0.},  // reference position
            hub_node->x,   // hub initial position
            hub_node->x,   // nacelle initial position
            {
                build_blade_config(0),  // Blade 1 config
                build_blade_config(1),  // Blade 2 config
                build_blade_config(2),  // Blade 3 config
            }
        ),
    };

    // Simulation controls
    util::SimulationControls sc;
    sc.aerodyn_input = "IEA-15-240-RWT_AeroDyn15.dat";
    sc.inflowwind_input = "IEA-15-240-RWT_InflowFile.dat";
    sc.time_step = step_size;
    sc.max_time = t_end;
    sc.total_elapsed_time = 0.;
    sc.n_time_steps = num_steps;
    sc.output_time_step = step_size;
    sc.debug_level = static_cast<int>(util::SimulationControls::DebugLevel::kAll);
    sc.transpose_DCM = false;

    // VTK settings
    util::VTKSettings vtk_settings;
    vtk_settings.write_vtk = 2;  // Animation
    vtk_settings.vtk_type = 2;   // Lines
    vtk_settings.vtk_nacelle_dimensions = {-2.5f, -2.5f, 0.f, 10.f, 5.f, 5.f};
    vtk_settings.vtk_hub_radius = static_cast<float>(hub_radius);

    util::AeroDynInflowLibrary adi(
        adi_shared_lib_path, util::ErrorHandling{}, util::FluidProperties{},
        util::EnvironmentalConditions{}, sc, vtk_settings
    );

    adi.Initialize(turbine_configs);

    //--------------------------------------------------------------------------
    // State
    //--------------------------------------------------------------------------

    auto state = model.CreateState();
    auto host_state_x = Kokkos::create_mirror(state.x);
    auto host_state_v = Kokkos::create_mirror(state.v);
    auto host_state_vd = Kokkos::create_mirror(state.vd);

    //--------------------------------------------------------------------------
    // Solver
    //--------------------------------------------------------------------------

    auto parameters = StepParameters(is_dynamic_solve, max_iter, step_size, rho_inf);
    auto solver = Solver(
        state.ID, beams.num_nodes_per_element, beams.node_state_indices, constraints.num_dofs,
        constraints.type, constraints.base_node_index, constraints.target_node_index,
        constraints.row_range
    );

    // Remove output directory for writing step data
    const std::filesystem::path step_dir("steps/IEA15RotorAeroController");
    RemoveDirectoryWithRetries(step_dir);
    std::filesystem::create_directory(step_dir.parent_path());
    std::filesystem::create_directory(step_dir);

    // Initialize rotor speed
    auto rotor_speed = rotor_speed_init;
    auto azimuth = azimuth_init;

    // Perform time steps and check for convergence within max_iter iterations
    for (size_t i = 0; i < num_steps; ++i) {
        // Write VTK output to file
        auto tmp = std::to_string(i);
        BeamsWriteVTK(
            beams,
            step_dir / (std::string("step_") + std::string(4 - tmp.size(), '0') + tmp + ".vtu")
        );

        // Get current time and next time
        const auto current_time{step_size * static_cast<double>(i)};
        const auto next_time{step_size * static_cast<double>(i + 1)};

        // Copy state matrices from device to host
        Kokkos::deep_copy(host_state_x, state.x);
        Kokkos::deep_copy(host_state_v, state.v);
        Kokkos::deep_copy(host_state_vd, state.vd);

        auto& turbine = adi.turbines[0];

        // Set rotor motion for current time
        turbine.hub.SetValues(
            0,
            Array_7{
                host_state_x(hub_node->ID, 0),
                host_state_x(hub_node->ID, 1),
                host_state_x(hub_node->ID, 2),
                host_state_x(hub_node->ID, 3),
                host_state_x(hub_node->ID, 4),
                host_state_x(hub_node->ID, 5),
                host_state_x(hub_node->ID, 6),
            },
            Array_6{
                host_state_v(hub_node->ID, 0),
                host_state_v(hub_node->ID, 1),
                host_state_v(hub_node->ID, 2),
                host_state_v(hub_node->ID, 3),
                host_state_v(hub_node->ID, 4),
                host_state_v(hub_node->ID, 5),
            },
            Array_6{
                host_state_vd(hub_node->ID, 0),
                host_state_vd(hub_node->ID, 1),
                host_state_vd(hub_node->ID, 2),
                host_state_vd(hub_node->ID, 3),
                host_state_vd(hub_node->ID, 4),
                host_state_vd(hub_node->ID, 5),
            }
        );

        // Nacelle is same as hub
        turbine.nacelle.position[0] = turbine.hub.position[0];
        turbine.nacelle.orientation[0] = turbine.hub.orientation[0];
        turbine.nacelle.velocity[0] = turbine.hub.velocity[0];
        turbine.nacelle.acceleration[0] = turbine.hub.acceleration[0];

        for (size_t j = 0; j < n_blades; ++j) {
            for (size_t k = 0; k < beam_elems[j].nodes.size(); ++k) {
                turbine.SetBladeNodeValues(
                    j, k,
                    Array_7{
                        host_state_x(beam_elems[j].nodes[k].node.ID, 0),
                        host_state_x(beam_elems[j].nodes[k].node.ID, 1),
                        host_state_x(beam_elems[j].nodes[k].node.ID, 2),
                        host_state_x(beam_elems[j].nodes[k].node.ID, 3),
                        host_state_x(beam_elems[j].nodes[k].node.ID, 4),
                        host_state_x(beam_elems[j].nodes[k].node.ID, 5),
                        host_state_x(beam_elems[j].nodes[k].node.ID, 6),
                    },
                    Array_6{
                        host_state_v(beam_elems[j].nodes[k].node.ID, 0),
                        host_state_v(beam_elems[j].nodes[k].node.ID, 1),
                        host_state_v(beam_elems[j].nodes[k].node.ID, 2),
                        host_state_v(beam_elems[j].nodes[k].node.ID, 3),
                        host_state_v(beam_elems[j].nodes[k].node.ID, 4),
                        host_state_v(beam_elems[j].nodes[k].node.ID, 5),
                    },
                    Array_6{
                        host_state_vd(beam_elems[j].nodes[k].node.ID, 0),
                        host_state_vd(beam_elems[j].nodes[k].node.ID, 1),
                        host_state_vd(beam_elems[j].nodes[k].node.ID, 2),
                        host_state_vd(beam_elems[j].nodes[k].node.ID, 3),
                        host_state_vd(beam_elems[j].nodes[k].node.ID, 4),
                        host_state_vd(beam_elems[j].nodes[k].node.ID, 5),
                    }
                );

                // Root node is same as first node of beam
                if (k == 0) {
                    auto point_num = turbine.node_indices_by_blade[j][k];
                    turbine.blade_roots.position[j] = turbine.blade_nodes.position[point_num];
                    turbine.blade_roots.orientation[j] = turbine.blade_nodes.orientation[point_num];
                    turbine.blade_roots.velocity[j] = turbine.blade_nodes.velocity[point_num];
                    turbine.blade_roots.acceleration[j] =
                        turbine.blade_nodes.acceleration[point_num];
                }
            }
        }
        adi.SetupRotorMotion();

        // Advance ADI library from current to next time
        adi.UpdateStates(current_time, next_time);

        adi.CalculateOutput(next_time);

        // Set controller inputs and call controller to get commands for this step
        const auto generator_speed = rotor_speed * gear_box_ratio;
        const auto generator_power = generator_speed * torque_actual;
        controller.io.status = 1;               // Subsequent call
        controller.io.time = current_time;      // Current time (seconds)
        controller.io.azimuth_angle = azimuth;  // Current azimuth angle (rad)
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

        // Predict new state,
        auto converged = Step(parameters, solver, beams, state, constraints);
        EXPECT_EQ(converged, true);

        // Update rotor speed
        rotor_speed = constraints.host_output(azimuth_constraint->ID, 1);
        azimuth = constraints.host_output(azimuth_constraint->ID, 0) + azimuth_init;
    }
}

}  // namespace openturbine::tests
