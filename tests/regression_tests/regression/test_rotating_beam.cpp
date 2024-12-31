#include <filesystem>
#include <fstream>
#include <initializer_list>
#include <iostream>

#include <gtest/gtest.h>

#include "test_utilities.hpp"

#ifdef OpenTurbine_ENABLE_VTK
#include "vtkout.hpp"
#endif

#include "src/dof_management/assemble_node_freedom_allocation_table.hpp"
#include "src/dof_management/compute_node_freedom_map_table.hpp"
#include "src/dof_management/create_constraint_freedom_table.hpp"
#include "src/dof_management/create_element_freedom_table.hpp"
#include "src/elements/beams/beam_element.hpp"
#include "src/elements/beams/beam_node.hpp"
#include "src/elements/beams/beam_section.hpp"
#include "src/elements/beams/beams.hpp"
#include "src/elements/beams/beams_input.hpp"
#include "src/elements/beams/create_beams.hpp"
#include "src/elements/elements.hpp"
#include "src/elements/masses/create_masses.hpp"
#include "src/model/model.hpp"
#include "src/solver/solver.hpp"
#include "src/state/state.hpp"
#include "src/step/step.hpp"
#include "src/types.hpp"

using Array_7 = std::array<double, 7>;

namespace openturbine::tests {

template <typename T>
void WriteMatrixToFile(const std::vector<std::vector<T>>& data, const std::string& filename) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Unable to open file: " << filename << "\n";
        return;
    }
    for (const auto& innerVector : data) {
        for (const auto& element : innerVector) {
            file << element << ",";
        }
        file << "\n";
    }
    file.close();
}

// Mass matrix for uniform composite beam section
constexpr auto mass_matrix = std::array{
    std::array{8.538e-2, 0., 0., 0., 0., 0.},    //
    std::array{0., 8.538e-2, 0., 0., 0., 0.},    //
    std::array{0., 0., 8.538e-2, 0., 0., 0.},    //
    std::array{0., 0., 0., 1.4433e-2, 0., 0.},   //
    std::array{0., 0., 0., 0., 0.40972e-2, 0.},  //
    std::array{0., 0., 0., 0., 0., 1.0336e-2},
};

// create a unity mass matrix
constexpr auto mass_matrix_unity = std::array{
    std::array{1., 0., 0., 0., 0., 0.},  //
    std::array{0., 1., 0., 0., 0., 0.},  //
    std::array{0., 0., 1., 0., 0., 0.},  //
    std::array{0., 0., 0., 1., 0., 0.},  //
    std::array{0., 0., 0., 0., 1., 0.},  //
    std::array{0., 0., 0., 0., 0., 1.},
};

// Stiffness matrix for uniform composite beam section
constexpr auto stiffness_matrix = std::array{
    std::array{1368.17e3, 0., 0., 0., 0., 0.},
    std::array{0., 88.56e3, 0., 0., 0., 0.},
    std::array{0., 0., 38.78e3, 0., 0., 0.},
    std::array{0., 0., 0., 16.9600e3, 17.6100e3, -0.3510e3},
    std::array{0., 0., 0., 17.6100e3, 59.1200e3, -0.3700e3},
    std::array{0., 0., 0., -0.3510e3, -0.3700e3, 141.470e3},
};

// create a unit stiffness matrix
constexpr auto stiffness_matrix_unity = std::array{
    std::array{1., 0., 0., 0., 0., 0.},  //
    std::array{0., 1., 0., 0., 0., 0.},  //
    std::array{0., 0., 1., 0., 0., 0.},  //
    std::array{0., 0., 0., 1., 0., 0.},  //
    std::array{0., 0., 0., 0., 1., 0.},  //
    std::array{0., 0., 0., 0., 0., 1.},
};

// Node locations (GLL quadrature)
const auto node_s = std::vector{
    0., 0.11747233803526763, 0.35738424175967748, 0.64261575824032247, 0.88252766196473242, 1.
};

// Element quadrature
const auto quadrature = BeamQuadrature{
    {-0.9491079123427585, 0.1294849661688697},  {-0.7415311855993943, 0.27970539148927664},
    {-0.40584515137739696, 0.3818300505051189}, {6.123233995736766e-17, 0.4179591836734694},
    {0.4058451513773971, 0.3818300505051189},   {0.7415311855993945, 0.27970539148927664},
    {0.9491079123427585, 0.1294849661688697},
};

const auto sections = std::vector{
    BeamSection(0., mass_matrix, stiffness_matrix),
    BeamSection(1., mass_matrix, stiffness_matrix),
};

const auto sections_unity = std::vector{
    BeamSection(0., mass_matrix_unity, stiffness_matrix_unity),
    BeamSection(1., mass_matrix_unity, stiffness_matrix_unity),
};

TEST(RotatingBeamTest, StepConvergence) {
    auto model = Model();

    // Gravity vector
    constexpr auto gravity = std::array{0., 0., 0.};

    // Build vector of nodes (straight along x axis, no rotation)
    // Calculate displacement, velocity, acceleration assuming a
    // 0.1 rad/s angular velocity around the z axis
    const double omega = 0.1;
    std::vector<BeamNode> nodes;
    std::transform(std::cbegin(node_s), std::cend(node_s), std::back_inserter(nodes), [&](auto s) {
        const auto x = 10 * s + 2.;
        return BeamNode(
            s, *model.AddNode(
                   {x, 0., 0., 1., 0., 0., 0.},        // Position
                   {0., 0., 0., 1., 0., 0., 0.},       // Displacement
                   {0., x * omega, 0., 0., 0., omega}  // Velocity
               )
        );
    });

    // Define beam initialization
    const auto beams_input = BeamsInput(
        {
            BeamElement(nodes, sections, quadrature),
        },
        gravity
    );

    // Initialize beams from element inputs
    auto beams = CreateBeams(beams_input);

    // No Masses
    const auto masses_input = MassesInput({}, gravity);
    auto masses = CreateMasses(masses_input);

    // Create elements from beams
    auto elements = Elements{beams, masses};

    // Constraint inputs
    model.AddPrescribedBC(model.GetNode(0));

    // Solution parameters
    const bool is_dynamic_solve(true);
    const int max_iter(5);
    const double step_size(0.01);  // seconds
    const double rho_inf(0.9);

    // Create solver
    auto parameters = StepParameters(is_dynamic_solve, max_iter, step_size, rho_inf);
    auto constraints = Constraints(model.GetConstraints());
    auto state = model.CreateState();
    assemble_node_freedom_allocation_table(state, elements, constraints);
    compute_node_freedom_map_table(state);
    create_element_freedom_table(elements, state);
    create_constraint_freedom_table(constraints, state);
    auto solver = Solver(
        state.ID, state.node_freedom_allocation_table, state.node_freedom_map_table,
        elements.NumberOfNodesPerElement(), elements.NodeStateIndices(), constraints.num_dofs,
        constraints.type, constraints.base_node_freedom_table, constraints.target_node_freedom_table,
        constraints.row_range
    );

    // Perform 10 time steps and check for convergence within max_iter iterations
    for (int i = 0; i < 10; ++i) {
        // Set constraint displacement
        const auto q = RotationVectorToQuaternion({0., 0., omega * step_size * (i + 1)});
        constraints.UpdateDisplacement(0, {0., 0., 0., q[0], q[1], q[2], q[3]});
        const auto converged = Step(parameters, solver, elements, state, constraints);
        EXPECT_EQ(converged, true);
    }

    expect_kokkos_view_2D_equal(
        state.q,
        {
            {-0.000099661884299369481, 0.019999672917628962, -3.6608854058480302E-25,
             0.99998750002604175, -1.5971376141505654E-26, 3.1592454262792375E-25,
             0.004999979166692714},
            {-0.00015838391157346692, 0.031746709275713193, -2.8155520815870626E-13,
             0.99998750002143066, 2.7244338869052949E-12, 1.989181042516661E-12,
             0.0049999800888738608},
            {-0.00027859681974392133, 0.055737500699772298, 2.815269319303426E-12,
             0.9999875000205457, 7.3510877107173739E-12, 1.0550370096863904E-12,
             0.0049999802658924715},
            {-0.00042131446700509681, 0.08426017738413949, 8.2854411551089936E-12,
             0.99998750002161218, 3.7252296525466957E-11, -5.26890056047209E-14,
             0.0049999800525935617},
            {-0.00054093210652801399, 0.10825097509997549, -9.3934322245617647E-12,
             0.99998750002142056, 4.0321076018153484E-11, 5.2579938812420674E-12,
             0.0049999800909203019},
            {-0.00059944528351138049, 0.11999801747595988, -2.6207280972097857E-11,
             0.99998750002237801, 3.4435006114567926E-11, 6.4250095159262128E-12,
             0.0049999798994432168},
        }
    );
}

inline void CreateTwoBeamSolverWithSameBeamsAndStep() {
    // Create model for managing nodes and constraints
    auto model = Model();

    // Gravity vector
    constexpr auto gravity = std::array{0., 0., 0.};

    // Rotor angular velocity in rad/s
    constexpr auto omega = std::array{0., 0., 1.};

    // Build vector of nodes (straight along x axis, no rotation)
    // Calculate displacement, velocity, acceleration assuming a
    // 0.1 rad/s angular velocity around the z axis
    constexpr auto num_blades = 2U;
    constexpr auto blade_list = std::array<int, num_blades>{};
    std::vector<BeamElement> blade_elems;

    // Loop through blades
    std::transform(
        std::cbegin(blade_list), std::cend(blade_list), std::back_inserter(blade_elems),
        [&](size_t) {
            // Define root rotation
            constexpr auto q_root = std::array{1., 0., 0., 0.};

            // Declare list of element nodes
            std::vector<BeamNode> beam_nodes;
            std::transform(
                std::cbegin(node_s), std::cend(node_s), std::back_inserter(beam_nodes),
                [&](auto s) {
                    const auto pos = RotateVectorByQuaternion(q_root, {10. * s + 2., 0., 0.});
                    const auto v = CrossProduct(omega, pos);
                    return BeamNode(
                        s, *model.AddNode(
                               {pos[0], pos[1], pos[2], q_root[0], q_root[1], q_root[2], q_root[3]
                               },                                                // position
                               {0., 0., 0., 1., 0., 0., 0.},                     // displacement
                               {v[0], v[1], v[2], omega[0], omega[1], omega[2]}  // velocity
                           )
                    );
                }
            );
            // Set constraint nodes
            model.AddPrescribedBC(beam_nodes[0].node);

            return BeamElement(beam_nodes, sections, quadrature);
        }
    );

    // Define beam initialization
    const auto beams_input = BeamsInput(blade_elems, gravity);

    // Initialize beams from element inputs
    auto beams = CreateBeams(beams_input);

    // No Masses
    const auto masses_input = MassesInput({}, gravity);
    auto masses = CreateMasses(masses_input);

    // Create elements from beams
    auto elements = Elements{beams, masses};

    // Solution parameters
    const bool is_dynamic_solve(true);
    const int max_iter(1);
    const double step_size(0.01);  // seconds
    const double rho_inf(0.9);

    // Create solver with initial node state
    auto parameters = StepParameters(is_dynamic_solve, max_iter, step_size, rho_inf);
    auto constraints = Constraints(model.GetConstraints());
    auto state = model.CreateState();
    assemble_node_freedom_allocation_table(state, elements, constraints);
    compute_node_freedom_map_table(state);
    create_element_freedom_table(elements, state);
    create_constraint_freedom_table(constraints, state);
    auto solver = Solver(
        state.ID, state.node_freedom_allocation_table, state.node_freedom_map_table,
        elements.NumberOfNodesPerElement(), elements.NodeStateIndices(), constraints.num_dofs,
        constraints.type, constraints.base_node_freedom_table, constraints.target_node_freedom_table,
        constraints.row_range
    );

    // Calculate hub rotation for this time step
    const auto q_hub =
        RotationVectorToQuaternion({step_size * omega[0], step_size * omega[1], step_size * omega[2]}
        );

    // Define hub translation/rotation displacement
    const auto u_hub = std::array{0., 0., 0., q_hub[0], q_hub[1], q_hub[2], q_hub[3]};

    // Update constraint displacements
    for (auto j = 0U; j < constraints.num_constraints; ++j) {
        constraints.UpdateDisplacement(j, u_hub);
    }

    // Take step, don't check for convergence, the following tests check that
    // all the elements were assembled properly
    Step(parameters, solver, elements, state, constraints);

    auto n = solver.num_system_dofs / 2;
    auto m = constraints.num_dofs / 2;

    // Check that R vector is the same for both beams
    auto R = kokkos_view_1D_to_vector(solver.R);
    for (auto i = 0U; i < n; ++i) {
        EXPECT_NEAR(R[i], R[n + i], 1.e-10);
    }

    // Check that Phi vector is the same for both beams
    auto Phi = kokkos_view_2D_to_vector(constraints.residual_terms);
    for (auto i = 0U; i < m; ++i) {
        EXPECT_NEAR(Phi[0][i], Phi[1][i], 1.e-10);
    }
}

TEST(RotatingBeamTest, TwoBeam) {
    CreateTwoBeamSolverWithSameBeamsAndStep();
}

TEST(RotatingBeamTest, ThreeBladeRotor) {
    auto model = Model();

    // Gravity vector
    constexpr auto gravity = std::array{0., 0., 9.81};

    // Rotor angular velocity in rad/s
    const auto omega = std::array{0., 0., 1.};

    // Build vector of nodes (straight along x axis, no rotation)
    // Calculate displacement, velocity, acceleration assuming a
    // 1 rad/s angular velocity around the z axis
    constexpr auto num_blades = 3U;
    auto blade_list = std::array<int, num_blades>{};
    std::iota(std::begin(blade_list), std::end(blade_list), 0);

    std::vector<BeamElement> blade_elems;
    std::transform(
        std::cbegin(blade_list), std::cend(blade_list), std::back_inserter(blade_elems),
        [&](auto i) {
            // Define root rotation
            const auto q_root = RotationVectorToQuaternion({0., 0., 2. * M_PI * i / num_blades});

            // Declare list of element nodes
            std::vector<BeamNode> beam_nodes;
            std::transform(
                std::cbegin(node_s), std::cend(node_s), std::back_inserter(beam_nodes),
                [&](auto s) {
                    const auto pos = RotateVectorByQuaternion(q_root, {10. * s + 2., 0., 0.});
                    auto v = CrossProduct(omega, pos);
                    return BeamNode(
                        s, *model.AddNode(
                               {pos[0], pos[1], pos[2], q_root[0], q_root[1], q_root[2], q_root[3]
                               },                                                // position
                               {0., 0., 0., 1., 0., 0., 0.},                     // displacement
                               {v[0], v[1], v[2], omega[0], omega[1], omega[2]}  // velocity
                           )
                    );
                }
            );

            // Set constraint nodes
            model.AddPrescribedBC(beam_nodes[0].node);

            return BeamElement(beam_nodes, sections, quadrature);
        }
    );

    // Define beam initialization
    const auto beams_input = BeamsInput(blade_elems, gravity);

    // Initialize beams from element inputs
    auto beams = CreateBeams(beams_input);

    // No Masses
    const auto masses_input = MassesInput({}, gravity);
    auto masses = CreateMasses(masses_input);

    // Create elements from beams
    auto elements = Elements{beams, masses};

    // Solution parameters
    const bool is_dynamic_solve(true);
    const int max_iter(4);
    const double step_size(0.01);  // seconds
    const double rho_inf(0.9);
    const double t_end(0.1);
    const auto num_steps = static_cast<size_t>(std::floor(t_end / step_size + 1.0));

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
    auto state = model.CreateState();
    assemble_node_freedom_allocation_table(state, elements, constraints);
    compute_node_freedom_map_table(state);
    create_element_freedom_table(elements, state);
    create_constraint_freedom_table(constraints, state);
    auto solver = Solver(
        state.ID, state.node_freedom_allocation_table, state.node_freedom_map_table,
        elements.NumberOfNodesPerElement(), elements.NodeStateIndices(), constraints.num_dofs,
        constraints.type, constraints.base_node_freedom_table, constraints.target_node_freedom_table,
        constraints.row_range
    );

    // Perform time steps and check for convergence within max_iter iterations
    for (auto i = 0U; i < num_steps; ++i) {
        // Calculate hub rotation for this time step
        const auto q_hub = RotationVectorToQuaternion(
            {step_size * (i + 1) * omega[0], step_size * (i + 1) * omega[1],
             step_size * (i + 1) * omega[2]}
        );

        // Define hub translation/rotation displacement
        const auto u_hub = std::array{0., 0., 0., q_hub[0], q_hub[1], q_hub[2], q_hub[3]};

        // Update constraint displacements
        for (auto j = 0U; j < constraints.num_constraints; ++j) {
            constraints.UpdateDisplacement(j, u_hub);
        }

        // Take step
        auto converged = Step(parameters, solver, elements, state, constraints);

        // Verify that step converged
        EXPECT_EQ(converged, true);
    }
}

TEST(RotatingBeamTest, MasslessConstraints) {
    auto model = Model();

    // Gravity vector
    constexpr auto gravity = std::array{0., 0., 0.};

    // Build vector of nodes (straight along x axis, no rotation)
    // Calculate displacement, velocity, acceleration assuming a
    // 0.1 rad/s angular velocity around the z axis
    const double omega = 0.1;
    std::vector<BeamNode> beam_nodes;
    std::transform(
        std::cbegin(node_s), std::cend(node_s), std::back_inserter(beam_nodes),
        [&](auto s) {
            auto x = 10 * s + 2.;
            return BeamNode(
                s, *model.AddNode(
                       {x, 0., 0., 1., 0., 0., 0.},        // position
                       {0., 0., 0., 1., 0., 0., 0.},       // displacement
                       {0., x * omega, 0., 0., 0., omega}  // velocity
                   )
            );
        }
    );

    // Define beam initialization
    const auto beams_input = BeamsInput({BeamElement(beam_nodes, sections, quadrature)}, gravity);

    // Initialize beams from element inputs
    auto beams = CreateBeams(beams_input);

    // No Masses
    const auto masses_input = MassesInput({}, gravity);
    auto masses = CreateMasses(masses_input);

    // Create elements from beams
    auto elements = Elements{beams, masses};

    // Add hub node and associated constraints
    auto hub_node = model.AddNode({0., 0., 0., 1., 0., 0., 0.});
    model.AddRigidJointConstraint(*hub_node, model.GetNode(0));
    auto hub_bc = model.AddPrescribedBC(*hub_node);

    // Solution parameters
    const bool is_dynamic_solve(true);
    const int max_iter(5);
    const double step_size(0.01);  // seconds
    const double rho_inf(0.9);

    // Create solver
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
    auto state = model.CreateState();
    assemble_node_freedom_allocation_table(state, elements, constraints);
    compute_node_freedom_map_table(state);
    create_element_freedom_table(elements, state);
    create_constraint_freedom_table(constraints, state);
    auto solver = Solver(
        state.ID, state.node_freedom_allocation_table, state.node_freedom_map_table,
        elements.NumberOfNodesPerElement(), elements.NodeStateIndices(), constraints.num_dofs,
        constraints.type, constraints.base_node_freedom_table, constraints.target_node_freedom_table,
        constraints.row_range
    );

    // Perform 10 time steps and check for convergence within max_iter iterations
    for (int i = 0; i < 10; ++i) {
        // Set constraint displacement
        const auto q = RotationVectorToQuaternion({0., 0., omega * step_size * (i + 1)});
        constraints.UpdateDisplacement(hub_bc->ID, {0., 0., 0., q[0], q[1], q[2], q[3]});
        const auto converged = Step(parameters, solver, elements, state, constraints);
        EXPECT_EQ(converged, true);
    }

    expect_kokkos_view_2D_equal(
        state.q,
        {{-0.000099661884299369481, 0.019999672917628962, -3.6608854058480302E-25,
          0.99998750002604175, -1.5971376141505654E-26, 3.1592454262792375E-25,
          0.004999979166692714},
         {-0.00015838391157346692, 0.031746709275713193, -2.8155520815870626E-13,
          0.99998750002143066, 2.7244338869052949E-12, 1.989181042516661E-12, 0.0049999800888738608},
         {-0.00027859681974392133, 0.055737500699772298, 2.815269319303426E-12, 0.9999875000205457,
          7.3510877107173739E-12, 1.0550370096863904E-12, 0.0049999802658924715},
         {-0.00042131446700509681, 0.08426017738413949, 8.2854411551089936E-12, 0.99998750002161218,
          3.7252296525466957E-11, -5.26890056047209E-14, 0.0049999800525935617},
         {-0.00054093210652801399, 0.10825097509997549, -9.3934322245617647E-12, 0.99998750002142056,
          4.0321076018153484E-11, 5.2579938812420674E-12, 0.0049999800909203019},
         {-0.00059944528351138049, 0.11999801747595988, -2.6207280972097857E-11, 0.99998750002237801,
          3.4435006114567926E-11, 6.4250095159262128E-12, 0.0049999798994432168},
         {0., 0., 0., 0.99998750002604219, 2.2269013449027429E-29, 1.884955233551297E-29,
          0.0049999791666927107}}
    );
}

TEST(RotatingBeamTest, RotationControlConstraint) {
    auto model = Model();

    // Gravity vector
    constexpr auto gravity = std::array{0., 0., 0.};

    // Build vector of nodes (straight along x axis, no rotation)
    std::vector<BeamNode> beam_nodes;
    std::transform(
        std::cbegin(node_s), std::cend(node_s), std::back_inserter(beam_nodes),
        [&](auto s) {
            return BeamNode(s, *model.AddNode({10 * s + 2., 0., 0., 1., 0., 0., 0.}));
        }
    );

    // Define beam initialization
    const auto beams_input = BeamsInput({BeamElement(beam_nodes, sections, quadrature)}, gravity);

    // Initialize beams from element inputs
    auto beams = CreateBeams(beams_input);

    // No Masses
    const auto masses_input = MassesInput({}, gravity);
    auto masses = CreateMasses(masses_input);

    // Create elements from beams
    auto elements = Elements{beams, masses};

    // Add hub node and associated constraints
    auto pitch = 0.;
    auto hub_node = model.AddNode({0., 0., 0., 1., 0., 0., 0.});
    model.AddRotationControl(*hub_node, beam_nodes[0].node, {1., 0., 0.}, &pitch);
    model.AddFixedBC(*hub_node);

    // Solution parameters
    const bool is_dynamic_solve(true);
    const int max_iter(5);
    const double step_size(0.01);  // seconds
    const double rho_inf(0.9);

    // Create solver
    auto parameters = StepParameters(is_dynamic_solve, max_iter, step_size, rho_inf);
    auto constraints = Constraints(model.GetConstraints());
    auto state = model.CreateState();
    assemble_node_freedom_allocation_table(state, elements, constraints);
    compute_node_freedom_map_table(state);
    create_element_freedom_table(elements, state);
    create_constraint_freedom_table(constraints, state);
    auto solver = Solver(
        state.ID, state.node_freedom_allocation_table, state.node_freedom_map_table,
        elements.NumberOfNodesPerElement(), elements.NodeStateIndices(), constraints.num_dofs,
        constraints.type, constraints.base_node_freedom_table, constraints.target_node_freedom_table,
        constraints.row_range
    );

    // Perform 10 time steps and check for convergence within max_iter iterations
    for (auto i = 0; i < 10; ++i) {
        const auto t = step_size * static_cast<double>(i + 1);
        // Set pitch
        pitch = t * M_PI / 2.;
        const auto converged = Step(parameters, solver, elements, state, constraints);
        EXPECT_EQ(converged, true);
    }

    // Check that remaining displacements are as expected
    expect_kokkos_view_2D_equal(
        state.q,
        {{-5.7690945215728628E-18, 2.0652319043893875E-18, -2.4953577261422928E-20,
          0.99691733356165013, 0.078459097906677058, 5.0946025505270351E-19, 3.3980589449407732E-19},
         {-2.9904494403209058E-7, 0.00014453413260242541, -0.00075720167307353353,
          0.99700858667767766, 0.077290133367305239, 0.00033527696532594382,
          0.000031054788779458509},
         {-0.000001465015511032517, 0.00057112956926573543, -0.0031674836974124104,
          0.99711128371780355, 0.075953862519306012, 0.00031282312872539991,
          0.000028155321221201416},
         {-0.0000020860700662326159, 0.00062108133928823466, -0.0035095491673744371,
          0.99720257290191016, 0.074745725742637242, -0.00032312502616037265,
          -0.000025908193827167231},
         {-0.0000043494810453724702, 0.00012641646939721653, -0.00026687208011275229,
          0.99728073405283557, 0.073693469884480778, -0.00063846060120356163,
          -0.000048492096870836455},
         {-0.0000058187577280119813, -0.00014174393267219577, 0.0015608892113182621,
          0.9972883429355086, 0.073590276246437658, -0.00065565396382017294,
          -0.000049038763777556399},
         {0, 0, 0, 1, 0, 0, 0}}
    );
}

TEST(RotatingBeamTest, CompoundRotationControlConstraint) {
    auto model = Model();

    // Gravity vector
    constexpr auto gravity = std::array{0., 0., 0.};

    // Build vector of nodes (straight along x axis, no rotation)
    std::vector<BeamNode> beam_nodes;
    std::transform(
        std::cbegin(node_s), std::cend(node_s), std::back_inserter(beam_nodes),
        [&](auto s) {
            return BeamNode(s, *model.AddNode({10 * s + 2., 0., 0., 1., 0., 0., 0.}));
        }
    );

    // Define beam initialization
    const auto beams_input = BeamsInput({BeamElement(beam_nodes, sections, quadrature)}, gravity);

    // Initialize beams from element inputs
    auto beams = CreateBeams(beams_input);

    // No Masses
    const auto masses_input = MassesInput({}, gravity);
    auto masses = CreateMasses(masses_input);

    // Create elements from beams
    auto elements = Elements{beams, masses};

    // Add hub node and associated constraints
    auto pitch = 0.;
    auto hub_node = model.AddNode({0., 0., 0., 1., 0., 0., 0.});
    model.AddRotationControl(*hub_node, beam_nodes[0].node, {1., 0., 0.}, &pitch);
    auto hub_bc = model.AddPrescribedBC(*hub_node);

    // Solution parameters
    const bool is_dynamic_solve(true);
    const int max_iter(5);
    const double step_size(0.01);  // seconds
    const double rho_inf(0.9);

    // Create solver
    auto parameters = StepParameters(is_dynamic_solve, max_iter, step_size, rho_inf);
    auto constraints = Constraints(model.GetConstraints());
    auto state = model.CreateState();
    assemble_node_freedom_allocation_table(state, elements, constraints);
    compute_node_freedom_map_table(state);
    create_element_freedom_table(elements, state);
    create_constraint_freedom_table(constraints, state);
    auto solver = Solver(
        state.ID, state.node_freedom_allocation_table, state.node_freedom_map_table,
        elements.NumberOfNodesPerElement(), elements.NodeStateIndices(), constraints.num_dofs,
        constraints.type, constraints.base_node_freedom_table, constraints.target_node_freedom_table,
        constraints.row_range
    );

    double azimuth = 0.;

    // Perform 10 time steps and check for convergence within max_iter iterations
    for (auto i = 0; i < 100; ++i) {
        const auto t = step_size * static_cast<double>(i + 1);
        pitch = t * M_PI / 2.;
        azimuth = 0.5 * t * M_PI / 2.;
        auto q = RotationVectorToQuaternion(Array_3{0., 0., azimuth});
        constraints.UpdateDisplacement(hub_bc->ID, {0., 0., 0., q[0], q[1], q[2], q[3]});
        const auto converged = Step(parameters, solver, elements, state, constraints);
        EXPECT_EQ(converged, true);
    }

    auto q = kokkos_view_2D_to_vector(state.q);
    auto rv = QuaternionToRotationVector(Array_4{q[0][3], q[0][4], q[0][5], q[0][6]});

    // Same as euler rotation xz [azimuth, pitch]
    EXPECT_NEAR(rv[0], 1.482189821649821, 1e-8);
    EXPECT_NEAR(rv[1], 0.61394312430788889, 1e-8);
    EXPECT_NEAR(rv[2], 0.61394312416734476, 1e-8);
}

TEST(RotatingBeamTest, RevoluteJointConstraint) {
    auto model = Model();

    // Gravity vector
    constexpr auto gravity = std::array{0., 0., 0.};

    // Build vector of nodes (straight along x axis, no rotation)
    // Calculate displacement, velocity, acceleration assuming a
    // 0.1 rad/s angular velocity around the z axis
    const double omega = 0.1;
    std::vector<BeamNode> beam_nodes;
    std::transform(
        std::cbegin(node_s), std::cend(node_s), std::back_inserter(beam_nodes),
        [&](auto s) {
            const auto x = 10 * s + 2.;
            return BeamNode(
                s, *model.AddNode(
                       {x, 0., 0., 1., 0., 0., 0.},        // position
                       {0., 0., 0., 1., 0., 0., 0.},       // displacement
                       {0., x * omega, 0., 0., 0., omega}  // velocity
                   )
            );
        }
    );

    // Define beam initialization
    const auto beams_input = BeamsInput({BeamElement(beam_nodes, sections, quadrature)}, gravity);

    // Initialize beams from element inputs
    auto beams = CreateBeams(beams_input);

    // No Masses
    const auto masses_input = MassesInput({}, gravity);
    auto masses = CreateMasses(masses_input);

    // Create elements from beams
    auto elements = Elements{beams, masses};

    // Add hub node and ground node
    auto hub_node = model.AddNode({0., 0., 0., 1., 0., 0., 0.});
    auto ground_node = model.AddNode({0, 0., -1., 1., 0., 0., 0.});

    // Add constraints
    model.AddFixedBC(*ground_node);  // Ground node is fixed

    // Revolute joint constraint
    auto torque = 0.;
    model.AddRevoluteJointConstraint(*ground_node, *hub_node, {0., 0., 0.}, &torque);

    model.AddRigidJointConstraint(*hub_node, beam_nodes[0].node);  // Hub node is rigidly connected

    // Solution parameters
    const bool is_dynamic_solve(true);
    const int max_iter(5);
    const double step_size(0.01);  // seconds
    const double rho_inf(0.9);

    // Create solver
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
    auto state = model.CreateState();
    assemble_node_freedom_allocation_table(state, elements, constraints);
    compute_node_freedom_map_table(state);
    create_element_freedom_table(elements, state);
    create_constraint_freedom_table(constraints, state);
    auto solver = Solver(
        state.ID, state.node_freedom_allocation_table, state.node_freedom_map_table,
        elements.NumberOfNodesPerElement(), elements.NodeStateIndices(), constraints.num_dofs,
        constraints.type, constraints.base_node_freedom_table, constraints.target_node_freedom_table,
        constraints.row_range
    );

#ifdef OpenTurbine_ENABLE_VTK
    UpdateSystemVariables(parameters, elements, state);
    RemoveDirectoryWithRetries("steps");
    std::filesystem::create_directory("steps");
    WriteVTKBeamsQP(elements.beams, "steps/step_0000.vtu");
#endif

    // Run 10 steps
    for (int i = 0; i < 5; ++i) {
        const auto converged = Step(parameters, solver, elements, state, constraints);
        EXPECT_EQ(converged, true);
#ifdef OpenTurbine_ENABLE_VTK
        auto tmp = std::to_string(i + 1);
        auto file_name = std::string("steps/step_") + std::string(4 - tmp.size(), '0') + tmp;
        WriteVTKBeamsQP(elements.beams, file_name + ".vtu");
#endif
    }

    expect_kokkos_view_2D_equal(
        state.q,
        {{-0.00002499992806604526, 0.009999954363364278, 7.0592758596667977E-26, 0.99999687500410894,
          8.0588517542554213E-29, 5.3983736084468692E-26, 0.0024999964033195574},
         {-0.000039681188471502004, 0.015873544516367064, 4.8213624609892645E-14,
          0.99999687500405054, -1.4117736696803771E-13, 4.0393896390903965E-14,
          0.0024999964266638514},
         {-0.000069668915568583335, 0.027869085208344892, 4.0193427070238621E-14,
          0.99999687500401824, -4.6940123973356305E-13, 7.4011787713126742E-14,
          0.0024999964396126765},
         {-0.00010532153768001662, 0.0421305961297474, 9.1114041847296216E-15, 0.99999687500407363,
          -2.4869679200969056E-13, 6.7014722734775103E-15, 0.0024999964174613749},
         {-0.00013530744593540648, 0.054126136607669233, -5.3259699489140029E-15,
          0.99999687500410672, 1.6008699820757778E-13, -6.0232179820024248E-14,
          0.0024999964042487837},
         {-0.00014999065003689889, 0.059999726696767466, 9.5097860930873942E-14, 0.9999968750041126,
          2.113602318183918E-13, -7.958169516703918E-14, 0.0024999964018790257},
         {0, 0, 0, 0.99999687500410894, 7.2192479817696718E-29, -2.7782381382812961E-27,
          0.0024999964033195574},
         {0, 0, 0, 1, 0, 0, 0}}
    );
}

void GeneratorTorqueWithAxisTilt(
    double tilt, const std::vector<double>& expected_azimuth_q,
    const std::vector<double>& expected_azimuth_vel,
    const std::vector<double>& expected_revolute_joint_output
) {
    auto model = Model();

    // Gravity vector - assume no gravity
    constexpr auto gravity = std::array{0., 0., 0.};

    // Calculate tilt about x axis as a quaternion
    auto node_tilt = RotationVectorToQuaternion({tilt, 0., 0.});

    // Build vector of nodes (straight along x axis, no rotation)
    std::vector<BeamNode> beam_nodes;
    std::transform(
        std::cbegin(node_s), std::cend(node_s), std::back_inserter(beam_nodes),
        [&](auto s) {
            const auto x = 10 * s + 2.;
            return BeamNode(
                s, *model.AddNode(
                       {x, std::sin(tilt), std::cos(tilt), node_tilt[0], node_tilt[1], node_tilt[2],
                        node_tilt[3]}  // position
                   )
            );
        }
    );

    // Define beam initialization
    const auto beams_input = BeamsInput({BeamElement(beam_nodes, sections, quadrature)}, gravity);

    // Initialize beams from element inputs
    auto beams = CreateBeams(beams_input);

    // No Masses
    const auto masses_input = MassesInput({}, gravity);
    auto masses = CreateMasses(masses_input);

    // Create elements from beams
    auto elements = Elements{beams, masses};

    // Add shaft base, azimuth, and hub nodes as massless points
    auto shaft_base = model.AddNode({0, 0., 0., 1., 0., 0., 0.});
    auto azimuth = model.AddNode({0, 0, 0, 1., 0., 0., 0.});
    auto hub = model.AddNode({0, std::sin(tilt), std::cos(tilt), 1., 0., 0., 0.});

    // Add constraints between the nodes to simulate a rotor with a generator
    model.AddFixedBC(*shaft_base);  // Fixed shaft base

    // Add torque to the azimuth node to simulate generator torque
    auto torque = 100.;
    auto shaft_rj = model.AddRevoluteJointConstraint(  // Azimuth can rotate around shaft base
        *shaft_base, *azimuth, {0., std::sin(tilt), std::cos(tilt)}, &torque
    );

    model.AddRigidJointConstraint(*azimuth, *hub);            // Hub is rigidly attached to azimuth
    model.AddRigidJointConstraint(*hub, beam_nodes[0].node);  // Beam is rigidly attached to hub

    // Solution parameters
    const bool is_dynamic_solve(true);
    const int max_iter(5);
    const double step_size(0.01);  // seconds
    const double rho_inf(0.);

    // Create solver
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
    auto state = model.CreateState();
    assemble_node_freedom_allocation_table(state, elements, constraints);
    compute_node_freedom_map_table(state);
    create_element_freedom_table(elements, state);
    create_constraint_freedom_table(constraints, state);
    auto solver = Solver(
        state.ID, state.node_freedom_allocation_table, state.node_freedom_map_table,
        elements.NumberOfNodesPerElement(), elements.NodeStateIndices(), constraints.num_dofs,
        constraints.type, constraints.base_node_freedom_table, constraints.target_node_freedom_table,
        constraints.row_range
    );

#ifdef OpenTurbine_ENABLE_VTK
    UpdateSystemVariables(parameters, elements, state);
    RemoveDirectoryWithRetries("steps");
    std::filesystem::create_directory("steps");
    WriteVTKBeamsQP(beams, "steps/step_0000.vtu");
#endif

    // Run 10 steps
    for (int i = 0; i < 10; ++i) {
        const auto converged = Step(parameters, solver, elements, state, constraints);
        EXPECT_EQ(converged, true);
#ifdef OpenTurbine_ENABLE_VTK
        auto tmp = std::to_string(i + 1);
        auto file_name = std::string("steps/step_") + std::string(4 - tmp.size(), '0') + tmp;
        WriteVTKBeamsQP(beams, file_name + ".vtu");
#endif
    }

    // Check that the azimuth node has rotated by the expected amount
    auto azimuth_q = Kokkos::View<double[7]>("azimuth_q");
    Kokkos::deep_copy(azimuth_q, Kokkos::subview(state.q, azimuth->ID, Kokkos::ALL));
    expect_kokkos_view_1D_equal(azimuth_q, expected_azimuth_q);

    // Check the azimuth node angular velocity is as expected
    auto azimuth_vel = Kokkos::View<double[6]>("azimuth_vel");
    Kokkos::deep_copy(azimuth_vel, Kokkos::subview(state.v, azimuth->ID, Kokkos::ALL));
    expect_kokkos_view_1D_equal(azimuth_vel, expected_azimuth_vel);

    // Get revolute joint output
    auto revolute_joint_out = Kokkos::View<double[3]>("revolute_joint_out");
    Kokkos::deep_copy(
        revolute_joint_out, Kokkos::subview(constraints.output, shaft_rj->ID, Kokkos::ALL)
    );
    // Check output (azimuth, angular velocity, angular acceleration)
    expect_kokkos_view_1D_equal(revolute_joint_out, expected_revolute_joint_output);
}

TEST(RotatingBeamTest, GeneratorTorque_Tilt0) {
    GeneratorTorqueWithAxisTilt(
        0.,                                            // Shaft tilt
        {0., 0., 0., 0.99998634, 0., 0., -0.0052267},  // Azimuth node rotational displacement
        {0., 0., 0., 0., 0., -0.18978539},             // Azimuth node rotational velocity
        {-0.01045353, -0.18978539, -0.566558}          // Shaft angular rotation, velocity, accel
    );
}

TEST(RotatingBeamTest, GeneratorTorque_Tilt90) {
    GeneratorTorqueWithAxisTilt(
        M_PI / 2.,                                     // Shaft tilt
        {0., 0., 0., 0.99998634, 0., -0.0052267, 0.},  // Azimuth node rotational displacement
        {0., 0., 0., 0., -0.18978539, 0.},             // Azimuth node rotational velocity
        {-0.01045353, -0.18978539, -0.566560}          // Shaft angular rotation, velocity, accel
    );
}

}  // namespace openturbine::tests
