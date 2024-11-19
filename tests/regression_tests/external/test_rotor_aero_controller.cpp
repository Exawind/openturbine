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
#include "src/dof_management/assemble_node_freedom_allocation_table.hpp"
#include "src/dof_management/compute_node_freedom_map_table.hpp"
#include "src/dof_management/create_element_freedom_table.hpp"
#include "src/model/model.hpp"
#include "src/solver/solver.hpp"
#include "src/state/state.hpp"
#include "src/step/step.hpp"
#include "src/types.hpp"
#include "src/utilities/aerodynamics/aerodyn_inflow.hpp"
#include "src/utilities/controllers/discon.hpp"
#include "src/utilities/controllers/turbine_controller.hpp"
#include "src/vendor/dylib/dylib.hpp"
#include "tests/regression_tests/regression/iea15_rotor_data.hpp"
#include "tests/regression_tests/regression/test_utilities.hpp"

#ifdef OpenTurbine_ENABLE_VTK
#include "tests/regression_tests/regression/vtkout.hpp"
#endif

namespace openturbine::tests {

constexpr bool use_node_loads = true;

Array_6 GetNodeData(const size_t index, const View_Nx6::HostMirror& state_matrix) {
    return Array_6{
        state_matrix(index, 0), state_matrix(index, 1), state_matrix(index, 2),
        state_matrix(index, 3), state_matrix(index, 4), state_matrix(index, 5),
    };
}

Array_7 GetNodeData(const size_t index, const View_Nx7::HostMirror& state_matrix) {
    return Array_7{
        state_matrix(index, 0), state_matrix(index, 1), state_matrix(index, 2),
        state_matrix(index, 3), state_matrix(index, 4), state_matrix(index, 5),
        state_matrix(index, 6),
    };
}

template <typename T1, typename T2>
Array_6 GetQPData(
    const size_t i, const size_t j, const T1& translation_data, const T2& rotation_data
) {
    return Array_6{
        translation_data(i, j, 0), translation_data(i, j, 1), translation_data(i, j, 2),
        rotation_data(i, j, 0),    rotation_data(i, j, 1),    rotation_data(i, j, 2),
    };
}

template <typename T1>
Array_7 GetQPData(const size_t i, const size_t j, const T1& position_matrix) {
    return Array_7{
        position_matrix(i, j, 0), position_matrix(i, j, 1), position_matrix(i, j, 2),
        position_matrix(i, j, 3), position_matrix(i, j, 4), position_matrix(i, j, 5),
        position_matrix(i, j, 6),
    };
}

template <
    typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7,
    typename T8>
void SetRotorMotion(
    util::TurbineData& turbine, const std::vector<BeamElement>& beam_elems,
    const std::vector<std::shared_ptr<Node>>& root_nodes,
    const std::shared_ptr<openturbine::Node>& hub_node, const T1& host_state_x,
    const T2& host_state_v, const T3& host_state_vd, const T4& host_qp_x, const T5& host_qp_u_dot,
    const T6& host_qp_omega, const T7& host_qp_u_ddot, const T8& host_qp_omega_dot
) {
    // Set rotor motion for current time
    turbine.SetHubMotion(
        GetNodeData(hub_node->ID, host_state_x), GetNodeData(hub_node->ID, host_state_v),
        GetNodeData(hub_node->ID, host_state_vd)
    );

    // Set rotor nacelle motion for current time (same as hub)
    turbine.SetNacelleMotion(
        GetNodeData(hub_node->ID, host_state_x), GetNodeData(hub_node->ID, host_state_v),
        GetNodeData(hub_node->ID, host_state_vd)
    );

    // Loop through blades and set blade root motion
    for (auto j = 0U; j < beam_elems.size(); ++j) {
        turbine.SetBladeRootMotion(
            j, GetNodeData(root_nodes[j]->ID, host_state_x),
            GetNodeData(root_nodes[j]->ID, host_state_v),
            GetNodeData(root_nodes[j]->ID, host_state_vd)
        );
    }

    if (use_node_loads) {
        // Loop through blade nodes
        for (auto j = 0U; j < beam_elems.size(); ++j) {
            for (auto k = 0U; k < beam_elems[j].nodes.size(); ++k) {
                turbine.SetBladeNodeMotion(
                    j, k, GetNodeData(beam_elems[j].nodes[k].node.ID, host_state_x),
                    GetNodeData(beam_elems[j].nodes[k].node.ID, host_state_v),
                    GetNodeData(beam_elems[j].nodes[k].node.ID, host_state_vd)
                );
            }
        }
    } else {
        // Loop through blade qps
        for (auto j = 0U; j < beam_elems.size(); ++j) {
            for (auto k = 0U; k < beam_elems[j].quadrature.size(); ++k) {
                turbine.SetBladeNodeMotion(
                    j, k, GetQPData(j, k, host_qp_x), GetQPData(j, k, host_qp_u_dot, host_qp_omega),
                    GetQPData(j, k, host_qp_u_ddot, host_qp_omega_dot)
                );
            }
        }
    }
}

template <typename T1, typename T2>
void SetAeroLoads(
    Beams& beams, const std::vector<BeamElement>& beam_elems, const util::TurbineData& turbine,
    const T1& host_node_FX, const T2& host_qp_Fe
) {
    if (use_node_loads) {
        for (size_t j = 0; j < beam_elems.size(); ++j) {
            for (size_t k = 0; k < beam_elems[j].nodes.size(); ++k) {
                const auto loads = turbine.GetBladeNodeLoad(j, k);
                for (size_t m = 0; m < 6U; ++m) {
                    host_node_FX(j, k, m) = loads[m];
                }
            }
        }
        Kokkos::deep_copy(beams.node_FX, host_node_FX);
    } else {
        for (size_t j = 0; j < beam_elems.size(); ++j) {
            for (size_t k = 0; k < beam_elems[j].quadrature.size(); ++k) {
                const auto loads = turbine.GetBladeNodeLoad(j, k);
                for (size_t m = 0; m < 6U; ++m) {
                    host_qp_Fe(j, k, m) = loads[m];
                }
            }
        }
        Kokkos::deep_copy(beams.qp_Fe, host_qp_Fe);
    }
}

TEST(Milestone, IEA15RotorAeroController) {
    // Conversions
    constexpr double rpm_to_radps{0.104719755};  // RPM to rad/s

    // Properties
    constexpr size_t n_blades{3};                            // Number of blades in rotor
    constexpr double azimuth_init{0.};                       // Azimuth angle (rad)
    constexpr double hub_height{150.};                       // Hub height (meters)
    constexpr double hub_radius{3.97};                       // Hub radius (meters)
    constexpr double gear_box_ratio{1.};                     // Gear box ratio (-)
    constexpr double rotor_speed_init{7.56 * rpm_to_radps};  // Rotor speed (rad/s)
    constexpr double hub_overhang{-50};                      // Hub overhang (meters)
    constexpr Array_3 shaft_axis = {1., 0., 0};              // Shaft along x-axis
    constexpr double hub_wind_speed_init{10.59};             // Hub height wind speed (m/s)
    constexpr double generator_power_init{15.0e6};           // Generator power (W)
    constexpr double blade_pitch_init{0.};                   // Initial blade pitch (rad)
    constexpr auto gravity = std::array{0., 0., -9.81};      // Gravity (m/s/s)

    // Controller parameters
    const std::string controller_shared_lib_path{"./ROSCO.dll"};
    const std::string controller_function_name{"DISCON"};
    const std::string controller_input_file_path{"./IEA-15-240-RWT/DISCON.IN"};
    const std::string controller_simulation_name{"./IEA-15-240-RWT"};

    // Aerodynamics and Inflow library
    const std::string adi_shared_lib_path{"./aerodyn_inflow_c_binding.dll"};
    const std::string aerodyn_input_path{"./IEA-15-240-RWT/AeroDyn15.dat"};
    const std::string inflowwind_input_path{"./IEA-15-240-RWT/InflowFile.dat"};

    // Solution parameters
    constexpr bool is_dynamic_solve{true};
    constexpr size_t max_iter{6};
    constexpr double step_size{0.01};  // seconds
    constexpr double rho_inf{0.0};
    constexpr double t_end{1.0};  // seconds
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
    controller.io.time = 0.;                               // Current time (seconds)
    controller.io.azimuth_angle = azimuth_init;            // Initial azimuth
    controller.io.pitch_blade1_actual = blade_pitch_init;  // Blade pitch (rad)
    controller.io.pitch_blade2_actual = blade_pitch_init;  // Blade pitch (rad)
    controller.io.pitch_blade3_actual = blade_pitch_init;  // Blade pitch (rad)
    controller.io.generator_speed_actual =
        rotor_speed_init * gear_box_ratio;  // Generator speed (rad/s)
    controller.io.generator_torque_actual =
        generator_power_init / (rotor_speed_init * gear_box_ratio);  // Generator torque
    controller.io.generator_power_actual = generator_power_init;     // Generator power (W)
    controller.io.rotor_speed_actual = rotor_speed_init;             // Rotor speed (rad/s)
    controller.io.horizontal_wind_speed = hub_wind_speed_init;       // Hub wind speed (m/s)

    // Signal first call
    controller.io.status = 0;

    // Make first call to controller
    controller.CallController();

    // Actual torque applied to shaft
    double torque_actual{controller.io.generator_torque_actual};
    double pitch_actual{blade_pitch_init};

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
    std::vector<Array_4> q_roots;
    for (size_t i = 0; i < n_blades; ++i) {
        q_roots.emplace_back(
            RotationVectorToQuaternion({d_theta * static_cast<double>(i) + azimuth_init, 0., 0.})
        );
    }

    constexpr Array_3 omega{
        rotor_speed_init * shaft_axis[0], rotor_speed_init * shaft_axis[1],
        rotor_speed_init * shaft_axis[2]
    };
    std::transform(
        std::cbegin(blade_list), std::cend(blade_list), std::back_inserter(beam_elems),
        [&](const size_t i) {
            // Define root rotation about x-axis
            const auto q_root = QuaternionCompose(q_roots[i], base_rot);

            // Declare vector of beam nodes
            std::vector<BeamNode> beam_nodes;

            // Loop through nodes in blade
            for (size_t j = 0; j < num_nodes; ++j) {
                // Calculate node position and orientation for this blade
                const auto rot = QuaternionCompose(q_root, node_rotation[j]);
                auto pos = RotateVectorByQuaternion(
                    q_root, {node_coords[j][0] + hub_radius, node_coords[j][1], node_coords[j][2]}
                );
                const auto v = CrossProduct(omega, pos);

                // Add hub overhang and hub height to position after calculating node velocity
                pos[0] += hub_overhang;
                pos[2] += hub_height;

                // Create beam node
                beam_nodes.emplace_back(
                    node_loc[j],
                    *model.AddNode(
                        {pos[0], pos[1], pos[2], rot[0], rot[1], rot[2], rot[3]},  // position
                        {0., 0., 0., 1., 0., 0., 0.},                              // displacement
                        {v[0], v[1], v[2], omega[0], omega[1], omega[2]}           // velocity
                    )
                );
            }

            // Add beam element
            return BeamElement(beam_nodes, material_sections, trapz_quadrature);
        }
    );

    // Vector of blade tip nodes
    std::vector<Node> tip_nodes;
    for (size_t i = 0; i < n_blades; ++i) {
        tip_nodes.emplace_back(beam_elems[i].nodes[beam_elems[i].nodes.size() - 1].node);
    }

    // Blade root nodes
    std::vector<std::shared_ptr<Node>> root_nodes;
    for (size_t i = 0; i < n_blades; ++i) {
        const auto q_root = QuaternionCompose(q_roots[i], base_rot);

        // Calculate node position and orientation for this blade
        auto pos = RotateVectorByQuaternion(
            q_root, {node_coords[0][0] + hub_radius, node_coords[0][1], node_coords[0][2]}
        );
        const auto v = CrossProduct(omega, pos);

        // Add hub overhang and hub height to position after calculating node velocity
        pos[0] += hub_overhang;
        pos[2] += hub_height;

        // If first node, add root node which doesn't include blade twist
        root_nodes.emplace_back(model.AddNode(
            {pos[0], pos[1], pos[2], q_root[0], q_root[1], q_root[2], q_root[3]},  // position
            {0., 0., 0., 1., 0., 0., 0.},                                          // displacement
            {v[0], v[1], v[2], omega[0], omega[1], omega[2]}                       // velocity
        ));
    }

    // Define beam initialization
    const auto beams_input = BeamsInput(beam_elems, gravity);

    // Initialize beams from element inputs
    auto beams = CreateBeams(beams_input);

    // Host mirror of beam external forces
    auto host_node_FX = Kokkos::create_mirror(beams.node_FX);

    //--------------------------------------------------------------------------
    // Rotor nodes
    //--------------------------------------------------------------------------

    auto shaft_base_node = model.AddNode({0., 0., hub_height, 1., 0., 0., 0.});
    auto azimuth_node = model.AddNode(
        {0., 0., hub_height, 1., 0., 0., 0.},       // Position
        {0., 0., 0., 1., 0., 0., 0.},               // Displacement
        {0., 0., 0., omega[0], omega[1], omega[2]}  // Velocity
    );
    auto hub_node = model.AddNode(
        {hub_overhang, 0., hub_height, 1., 0., 0., 0.},  // Position
        {0., 0., 0., 1., 0., 0., 0.},                    // Displacement
        {0., 0., 0., omega[0], omega[1], omega[2]}       // Velocity
    );

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
    for (size_t i = 0; i < n_blades; ++i) {
        // Calculate pitch axis from hub node to blade root node
        const Array_3 pitch_axis{
            hub_node->x[0] - root_nodes[i]->x[0],
            hub_node->x[1] - root_nodes[i]->x[1],
            hub_node->x[2] - root_nodes[i]->x[2],
        };

        // Add rotation control constraint between hub and root node
        model.AddRotationControl(*hub_node, *root_nodes[i], pitch_axis, &pitch_actual);

        // Add rigid constraint between root node and first blade node
        model.AddRigidJointConstraint(*root_nodes[i], beam_elems[i].nodes[0].node);
    }

    // Create constraints object
    auto constraints = Constraints(model.GetConstraints());

    //--------------------------------------------------------------------------
    // State
    //--------------------------------------------------------------------------

    // Create state
    auto state = model.CreateState();

    // Create mirrors for accessing node data
    auto host_state_x = Kokkos::create_mirror(state.x);
    auto host_state_q = Kokkos::create_mirror(state.q);
    auto host_state_v = Kokkos::create_mirror(state.v);
    auto host_state_vd = Kokkos::create_mirror(state.vd);

    //--------------------------------------------------------------------------
    // Solver
    //--------------------------------------------------------------------------

    // Create solver parameters
    auto parameters = StepParameters(is_dynamic_solve, max_iter, step_size, rho_inf);

    // Create solver
    assemble_node_freedom_allocation_table(state, beams, constraints);
    compute_node_freedom_map_table(state);
    create_element_freedom_table(beams, state);
    auto solver = Solver(
        state.ID, state.node_freedom_allocation_table, state.node_freedom_map_table,
        beams.num_nodes_per_element, beams.node_state_indices, constraints.num_dofs,
        constraints.type, constraints.base_node_index, constraints.target_node_index,
        constraints.row_range
    );

    // Transfer initial state to beams for writing output
    UpdateSystemVariables(parameters, beams, state);

    //--------------------------------------------------------------------------
    // AeroDyn / InflowWind library
    //--------------------------------------------------------------------------

    // Create mirrors for accessing qp data
    auto host_qp_x = Kokkos::create_mirror(beams.qp_x);
    auto host_qp_u_dot = Kokkos::create_mirror(beams.qp_u_dot);
    auto host_qp_omega = Kokkos::create_mirror(beams.qp_omega);
    auto host_qp_u_ddot = Kokkos::create_mirror(beams.qp_u_ddot);
    auto host_qp_omega_dot = Kokkos::create_mirror(beams.qp_omega_dot);
    auto host_qp_Fe = Kokkos::create_mirror(beams.qp_Fe);

    Kokkos::deep_copy(host_qp_x, beams.qp_x);
    Kokkos::deep_copy(host_qp_u_dot, beams.qp_u_dot);
    Kokkos::deep_copy(host_qp_omega, beams.qp_omega);
    Kokkos::deep_copy(host_qp_u_ddot, beams.qp_u_ddot);
    Kokkos::deep_copy(host_qp_omega_dot, beams.qp_omega_dot);

    // Create lambda for building blade configuration
    auto build_blade_config = [&](size_t i_blade) {
        std::vector<Array_7> mesh_positions;
        if (use_node_loads) {
            std::transform(
                beam_elems[i_blade].nodes.begin(), beam_elems[i_blade].nodes.end(),
                std::back_inserter(mesh_positions),
                [](const BeamNode& n) {
                    return n.node.x;
                }
            );
        } else {
            for (auto i_qp = 0U; i_qp < beam_elems[i_blade].quadrature.size(); ++i_qp) {
                mesh_positions.emplace_back(GetQPData(i_blade, i_qp, host_qp_x));
            }
        }
        return util::TurbineConfig::BladeInitialState{
            root_nodes[i_blade]->x,  // Root node
            mesh_positions,          // Blade nodes
        };
    };

    // Define turbine initial position
    const std::vector<util::TurbineConfig> turbine_configs{
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
    sc.aerodyn_input = aerodyn_input_path;
    sc.inflowwind_input = inflowwind_input_path;
    sc.time_step = step_size;
    sc.max_time = t_end;
    sc.total_elapsed_time = 0.;
    sc.n_time_steps = num_steps;
    sc.output_time_step = step_size;
    sc.point_load_output = use_node_loads;
    sc.debug_level = util::SimulationControls::DebugLevel::kNone;
    sc.transpose_DCM = false;
    sc.output_format = util::SimulationControls::OutputFormat::kNone;

    // VTK settings
    util::VTKSettings vtk_settings;
    vtk_settings.write_vtk = util::VTKSettings::WriteType::kNone;  // Animation
    vtk_settings.vtk_type = util::VTKSettings::OutputType::kLine;  // Lines
    vtk_settings.vtk_nacelle_dimensions = {-2.5F, -2.5F, 0.F, 10.F, 5.F, 5.F};
    vtk_settings.vtk_hub_radius = static_cast<float>(hub_radius);

    util::AeroDynInflowLibrary adi(
        adi_shared_lib_path, util::ErrorHandling{}, util::FluidProperties{},
        util::EnvironmentalConditions{}, sc, vtk_settings
    );

    // Remove the ADI vtk folder if outputting animation
    if (vtk_settings.write_vtk != util::VTKSettings::WriteType::kNone) {
        RemoveDirectoryWithRetries("vtk-ADI");
    }

    adi.Initialize(turbine_configs);

    //--------------------------------------------------------------------------
    // Output
    //--------------------------------------------------------------------------

    // Remove output directory for writing step data
    const std::filesystem::path step_dir("steps/IEA15RotorAeroController");
    RemoveDirectoryWithRetries(step_dir);
    std::filesystem::create_directory(step_dir.parent_path());
    std::filesystem::create_directory(step_dir);

    // Write output file header
    std::ofstream w("IEA15RotorAeroController.out");
    w << std::scientific;
    w << std::left;
    const std::vector<std::array<std::string, 2>> header_data{
        {"Time", "(s)"},         //
        {"ConvIter", "(-)"},     //
        {"ConvError", "(-)"},    //
        {"Azimuth", "(deg)"},    //
        {"BldPitch1", "(deg)"},  //
        {"GenSpeed", "(rpm)"},   //
        {"GenTq", "(kN-m)"},     //
        {"GenPwr", "(kW)"},      //
        {"B1TipTDxr", "(m)"},    //
        {"B2TipTDxr", "(m)"},    //
        {"B3TipTDxr", "(m)"},
    };
    for (const auto& item : header_data) {
        w << "\t" << item[0];
    }
    for (const auto& name : adi.channel_names) {
        w << "\t" << name;
    }
    w << "\n";
    for (const auto& item : header_data) {
        w << "\t" << item[1];
    }
    for (const auto& unit : adi.channel_units) {
        w << "\t" << unit;
    }
    w << "\n";
    w << std::right;

    //--------------------------------------------------------------------------
    // Time stepping
    //--------------------------------------------------------------------------

    // Initialize rotor speed
    auto rotor_speed = rotor_speed_init;
    auto azimuth = azimuth_init;

    // Perform time steps and check for convergence within max_iter iterations
    for (size_t i = 0; i < num_steps; ++i) {
        auto time_step_region = Kokkos::Profiling::ScopedRegion("Time Step");
#ifdef OpenTurbine_ENABLE_VTK
        auto tmp = std::to_string(i);
        tmp.insert(0, 5 - tmp.size(), '0');
        WriteVTKBeamsQP(beams, step_dir / (std::string("step_qp.") + tmp + ".vtu"));
        WriteVTKBeamsNodes(beams, step_dir / (std::string("step_node.") + tmp + ".vtu"));
#endif

        // Get current time and next time
        const auto current_time{step_size * static_cast<double>(i)};
        const auto next_time{step_size * static_cast<double>(i + 1)};

        {
            auto aerodyn_region = Kokkos::Profiling::ScopedRegion("AeroDyn");
            // Copy state matrices from device to host
            Kokkos::deep_copy(host_state_x, state.x);
            Kokkos::deep_copy(host_state_q, state.q);
            Kokkos::deep_copy(host_state_v, state.v);
            Kokkos::deep_copy(host_state_vd, state.vd);

            Kokkos::deep_copy(host_qp_x, beams.qp_x);
            Kokkos::deep_copy(host_qp_u_dot, beams.qp_u_dot);
            Kokkos::deep_copy(host_qp_omega, beams.qp_omega);
            Kokkos::deep_copy(host_qp_u_ddot, beams.qp_u_ddot);
            Kokkos::deep_copy(host_qp_omega_dot, beams.qp_omega_dot);

            // Set rotor motion from nodes or quadrature points
            SetRotorMotion(
                adi.turbines[0], beam_elems, root_nodes, hub_node, host_state_x, host_state_v,
                host_state_vd, host_qp_x, host_qp_u_dot, host_qp_omega, host_qp_u_ddot,
                host_qp_omega_dot
            );
            adi.SetRotorMotion();

            // Advance ADI library from current time to end of step
            adi.UpdateStates(current_time, next_time);

            // Calculate outputs and loads at the end of the step
            adi.CalculateOutput(next_time);

            // Copy aerodynamic loads to structure
            SetAeroLoads(beams, beam_elems, adi.turbines[0], host_node_FX, host_qp_Fe);
        }

        // Set controller inputs and call controller to get commands for this step
        const auto generator_speed = rotor_speed * gear_box_ratio;
        const auto generator_power = generator_speed * torque_actual;
        controller.io.status = 1;               // Subsequent call
        controller.io.time = current_time;      // Current time (seconds)
        controller.io.azimuth_angle = azimuth;  // Current azimuth angle (rad)
        controller.io.pitch_blade1_actual = pitch_actual;
        controller.io.pitch_blade2_actual = pitch_actual;
        controller.io.pitch_blade3_actual = pitch_actual;
        controller.io.rotor_speed_actual = rotor_speed;          // Rotor speed (rad/s)
        controller.io.generator_speed_actual = generator_speed;  // Generator speed (rad/s)
        controller.io.generator_power_actual = generator_power;  // Generator power (W)
        controller.io.generator_torque_actual = torque_actual;   // Generator torque (N-m)
        controller.io.horizontal_wind_speed =
            static_cast<double>(adi.turbines[0].hh_vel[0]);  // Hub wind speed (m/s)
        controller.CallController();

        // Update the generator torque and blade pitch
        torque_actual = controller.io.generator_torque_command;
        pitch_actual = controller.io.pitch_collective_command;

        // Write data to output file
        const auto conv_err{solver.convergence_err.empty() ? 0. : solver.convergence_err.back()};
        w << "\t" << current_time                                    // current time (s)
          << "\t" << solver.convergence_err.size()                   // num convergence iterations
          << "\t" << conv_err                                        // convergence error
          << "\t" << azimuth * 180. / M_PI                           // azimuth angle (deg)
          << "\t" << pitch_actual * 180. / M_PI                      // blade pitch (deg)
          << "\t" << generator_speed / rpm_to_radps                  //
          << "\t" << controller.io.generator_torque_command / 1000.  //
          << "\t" << controller.io.generator_power_actual / 1000.    //
          << "\t" << GetNodeData(tip_nodes[0].ID, host_state_q)[0]   // x displacement of tip nodes
          << "\t" << GetNodeData(tip_nodes[1].ID, host_state_q)[0]   // x displacement of tip nodes
          << "\t" << GetNodeData(tip_nodes[2].ID, host_state_q)[0];  // x displacement of tip nodes
        for (const auto& value : adi.channel_values) {
            w << "\t" << value;
        }
        w << "\n";

        // Predict state at end of step
        auto converged = Step(parameters, solver, beams, state, constraints);
        EXPECT_EQ(converged, true);

        // Update rotor azimuth and speed
        azimuth = constraints.host_output(azimuth_constraint->ID, 0) + azimuth_init;
        if (azimuth < 0) {
            azimuth += 2. * M_PI;
        }
        rotor_speed = constraints.host_output(azimuth_constraint->ID, 1);
    }
}

}  // namespace openturbine::tests
