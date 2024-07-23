#include <fstream>
#include <initializer_list>
#include <iostream>

#include <gtest/gtest.h>

#include "test_utilities.hpp"

#ifdef OTURB_ENABLE_VTK
#include "vtkout.hpp"
#endif

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

using Array_7 = std::array<double, 7>;

namespace openturbine::restruct_poc::tests {

template <typename T>
void WriteMatrixToFile(const std::vector<std::vector<T>>& data, const std::string& filename) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Unable to open file: " << filename << std::endl;
        return;
    }
    for (const auto& innerVector : data) {
        for (const auto& element : innerVector) {
            file << element << ",";
        }
        file << std::endl;
    }
    file.close();
}

// Mass matrix for uniform composite beam section
Array_6x6 mass_matrix = {{
    {8.538e-2, 0., 0., 0., 0., 0.},
    {0., 8.538e-2, 0., 0., 0., 0.},
    {0., 0., 8.538e-2, 0., 0., 0.},
    {0., 0., 0., 1.4433e-2, 0., 0.},
    {0., 0., 0., 0., 0.40972e-2, 0.},
    {0., 0., 0., 0., 0., 1.0336e-2},
}};

// Stiffness matrix for uniform composite beam section
Array_6x6 stiffness_matrix = {{
    {1368.17e3, 0., 0., 0., 0., 0.},
    {0., 88.56e3, 0., 0., 0., 0.},
    {0., 0., 38.78e3, 0., 0., 0.},
    {0., 0., 0., 16.9600e3, 17.6100e3, -0.3510e3},
    {0., 0., 0., 17.6100e3, 59.1200e3, -0.3700e3},
    {0., 0., 0., -0.3510e3, -0.3700e3, 141.470e3},
}};

// Node locations (GLL quadrature)
std::vector<double> node_s(
    {0., 0.11747233803526763, 0.35738424175967748, 0.64261575824032247, 0.88252766196473242, 1.}
);

// Element quadrature
BeamQuadrature quadrature{
    {-0.9491079123427585, 0.1294849661688697},  {-0.7415311855993943, 0.27970539148927664},
    {-0.40584515137739696, 0.3818300505051189}, {6.123233995736766e-17, 0.4179591836734694},
    {0.4058451513773971, 0.3818300505051189},   {0.7415311855993945, 0.27970539148927664},
    {0.9491079123427585, 0.1294849661688697},
};

std::vector<BeamSection> sections = {
    BeamSection(0., mass_matrix, stiffness_matrix),
    BeamSection(1., mass_matrix, stiffness_matrix),
};

TEST(RotatingBeamTest, StepConvergence) {
    auto model = Model_2();

    // Gravity vector
    std::array<double, 3> gravity = {0., 0., 0.};

    // Build vector of nodes (straight along x axis, no rotation)
    // Calculate displacement, velocity, acceleration assuming a
    // 0.1 rad/s angular velocity around the z axis
    const double omega = 0.1;
    std::vector<BeamNode> nodes;
    for (const double s : node_s) {
        auto x = 10 * s + 2.;
        auto node = model.AddNode(
            {x, 0., 0., 1., 0., 0., 0.},        // Position
            {0., 0., 0., 1., 0., 0., 0.},       // Displacement
            {0., x * omega, 0., 0., 0., omega}  // Velocity
        );
        nodes.push_back(BeamNode(s, *node));
    }

    // Define beam initialization
    BeamsInput beams_input(
        {
            BeamElement(nodes, sections, quadrature),
        },
        gravity
    );

    // Initialize beams from element inputs
    auto beams = CreateBeams(beams_input);

    // Constraint inputs
    // model.AddPrescribedBC(model.nodes[0]);
    model.AddPrescribedBC(*model.GetNode(0));

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

    Solver solver(
        is_dynamic_solve, max_iter, step_size, rho_inf, nodes_vector, constraints_vector, beams
    );

    // Perform 10 time steps and check for convergence within max_iter iterations
    for (int i = 0; i < 10; ++i) {
        // Set constraint displacement
        const auto q = RotationVectorToQuaternion({0., 0., omega * step_size * (i + 1)});
        solver.constraints.UpdateDisplacement(0, {0., 0., 0., q[0], q[1], q[2], q[3]});
        const auto converged = Step(solver, beams);
        EXPECT_EQ(converged, true);
    }

    expect_kokkos_view_2D_equal(
        solver.state.q,
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

TEST(RotatingBeamTest, TwoBeam) {
    // Create model for managing nodes and constraints
    auto model = Model();

    // Gravity vector
    std::array<double, 3> gravity = {0., 0., 0.};

    // Rotor angular velocity in rad/s
    const auto omega = std::array<double, 3>{0., 0., 1.};

    // Build vector of nodes (straight along x axis, no rotation)
    // Calculate displacement, velocity, acceleration assuming a
    // 0.1 rad/s angular velocity around the z axis
    const int num_blades = 2;
    std::vector<BeamElement> blade_elems;

    // Loop through blades
    for (int i = 0; i < num_blades; ++i) {
        // Define root rotation
        const auto q_root = std::array<double, 4>{1., 0., 0., 0.};

        // Declare list of element nodes
        std::vector<BeamNode> beam_nodes;

        // Loop through nodes
        for (const double s : node_s) {
            const auto pos = RotateVectorByQuaternion(q_root, {10. * s + 2., 0., 0.});
            auto v = CrossProduct(omega, pos);
            beam_nodes.push_back(BeamNode(
                s, model.AddNode(
                       {pos[0], pos[1], pos[2], q_root[0], q_root[1], q_root[2],
                        q_root[3]},                                      // position
                       {0., 0., 0., 1., 0., 0., 0.},                     // displacement
                       {v[0], v[1], v[2], omega[0], omega[1], omega[2]}  // velocity
                   )
            ));
        }

        // Add beam element
        blade_elems.push_back(BeamElement(beam_nodes, sections, quadrature));

        // Set constraint nodes
        model.AddPrescribedBC(beam_nodes[0].node);
    }

    // Define beam initialization
    BeamsInput beams_input(blade_elems, gravity);

    // Initialize beams from element inputs
    auto beams = CreateBeams(beams_input);

    // Solution parameters
    const bool is_dynamic_solve(true);
    const int max_iter(1);
    const double step_size(0.01);  // seconds
    const double rho_inf(0.9);

    // Create solver with initial node state
    Solver solver(
        is_dynamic_solve, max_iter, step_size, rho_inf, model.nodes, model.constraints, beams
    );

    // Calculate hub rotation for this time step
    const auto q_hub =
        RotationVectorToQuaternion({step_size * omega[0], step_size * omega[1], step_size * omega[2]}
        );

    // Define hub translation/rotation displacement
    Array_7 u_hub({0, 0, 0, q_hub[0], q_hub[1], q_hub[2], q_hub[3]});

    // Update constraint displacements
    for (int j = 0; j < solver.constraints.num; ++j) {
        solver.constraints.UpdateDisplacement(j, u_hub);
    }

    // Take step, don't check for convergence, the following tests check that
    // all the elements were assembled properly
    Step(solver, beams);

    auto n = solver.num_system_dofs / 2;
    auto m = solver.constraints.num_dofs / 2;

    // Check that St matrix is the same for both beams
    auto St = kokkos_view_2D_to_vector(solver.St);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            EXPECT_NEAR(St[i][j], St[n + i][n + j], 1.e-10);
        }
    }

    // Check that R vector is the same for both beams
    auto R = kokkos_view_1D_to_vector(solver.R);
    for (int i = 0; i < n; ++i) {
        EXPECT_NEAR(R[i], R[n + i], 1.e-10);
    }

    // Check that Phi vector is the same for both beams
    auto Phi = kokkos_view_1D_to_vector(solver.constraints.Phi);
    for (int i = 0; i < m; ++i) {
        EXPECT_NEAR(Phi[i], Phi[i + m], 1.e-10);
    }

    // Check that B matrix is the same for both beams
    auto B = kokkos_view_2D_to_vector(solver.constraints.B);
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            EXPECT_NEAR(B[i][j], B[i + m][j + n], 1.e-10);
        }
    }
}

TEST(RotatingBeamTest, ThreeBladeRotor) {
    auto model = Model();

    // Gravity vector
    Array_3 gravity = {0., 0., 9.81};

    // Rotor angular velocity in rad/s
    const auto omega = Array_3{0., 0., 1.};

    // Build vector of nodes (straight along x axis, no rotation)
    // Calculate displacement, velocity, acceleration assuming a
    // 1 rad/s angular velocity around the z axis
    const int num_blades = 3;
    std::vector<BeamElement> blade_elems;

    // Loop through blades
    for (int i = 0; i < num_blades; ++i) {
        // Define root rotation
        const auto q_root = RotationVectorToQuaternion({0., 0., 2. * M_PI * i / num_blades});

        // Declare list of element nodes
        std::vector<BeamNode> beam_nodes;

        // Loop through nodes
        for (const double s : node_s) {
            const auto pos = RotateVectorByQuaternion(q_root, {10. * s + 2., 0., 0.});
            auto v = CrossProduct(omega, pos);
            beam_nodes.push_back(BeamNode(
                s, model.AddNode(
                       {pos[0], pos[1], pos[2], q_root[0], q_root[1], q_root[2],
                        q_root[3]},                                      // position
                       {0., 0., 0., 1., 0., 0., 0.},                     // displacement
                       {v[0], v[1], v[2], omega[0], omega[1], omega[2]}  // velocity
                   )
            ));
        }

        // Add beam element
        blade_elems.push_back(BeamElement(beam_nodes, sections, quadrature));

        // Set constraint nodes
        model.AddPrescribedBC(beam_nodes[0].node);
    }

    // Define beam initialization
    BeamsInput beams_input(blade_elems, gravity);

    // Initialize beams from element inputs
    auto beams = CreateBeams(beams_input);

    // Solution parameters
    const bool is_dynamic_solve(true);
    const int max_iter(4);
    const double step_size(0.01);  // seconds
    const double rho_inf(0.9);
    const double t_end(0.1);
    const int num_steps(t_end / step_size + 1.0);

    // Create solver with initial node state
    Solver solver(
        is_dynamic_solve, max_iter, step_size, rho_inf, model.nodes, model.constraints, beams
    );

    // Perform time steps and check for convergence within max_iter iterations
    for (int i = 0; i < num_steps; ++i) {
        // Calculate hub rotation for this time step
        const auto q_hub = RotationVectorToQuaternion(
            {step_size * (i + 1) * omega[0], step_size * (i + 1) * omega[1],
             step_size * (i + 1) * omega[2]}
        );

        // Define hub translation/rotation displacement
        Array_7 u_hub({0, 0, 0, q_hub[0], q_hub[1], q_hub[2], q_hub[3]});

        // Update constraint displacements
        for (int j = 0; j < solver.constraints.num; ++j) {
            solver.constraints.UpdateDisplacement(j, u_hub);
        }

        // Take step
        auto converged = Step(solver, beams);

        // Verify that step converged
        EXPECT_EQ(converged, true);
    }
}

TEST(RotatingBeamTest, MasslessConstraints) {
    auto model = Model();

    // Gravity vector
    std::array<double, 3> gravity = {0., 0., 0.};

    // Build vector of nodes (straight along x axis, no rotation)
    // Calculate displacement, velocity, acceleration assuming a
    // 0.1 rad/s angular velocity around the z axis
    const double omega = 0.1;
    std::vector<BeamNode> beam_nodes;
    for (const double s : node_s) {
        auto x = 10 * s + 2.;
        beam_nodes.push_back(BeamNode(
            s, model.AddNode(
                   {x, 0., 0., 1., 0., 0., 0.},        // position
                   {0., 0., 0., 1., 0., 0., 0.},       // displacement
                   {0., x * omega, 0., 0., 0., omega}  // velocity
               )
        ));
    }

    // Define beam initialization
    BeamsInput beams_input({BeamElement(beam_nodes, sections, quadrature)}, gravity);

    // Initialize beams from element inputs
    auto beams = CreateBeams(beams_input);

    // Add hub node and associated constraints
    auto hub_node = model.AddNode({0., 0., 0., 1., 0., 0., 0.});
    model.AddRigidConstraint(hub_node, beam_nodes[0].node);
    auto hub_bc = model.AddPrescribedBC(hub_node);

    // Solution parameters
    const bool is_dynamic_solve(true);
    const int max_iter(5);
    const double step_size(0.01);  // seconds
    const double rho_inf(0.9);

    // Create solver
    Solver solver(
        is_dynamic_solve, max_iter, step_size, rho_inf, model.nodes, model.constraints, beams
    );

    // Perform 10 time steps and check for convergence within max_iter iterations
    for (int i = 0; i < 10; ++i) {
        // Set constraint displacement
        const auto q = RotationVectorToQuaternion({0., 0., omega * step_size * (i + 1)});
        solver.constraints.UpdateDisplacement(hub_bc.ID, {0., 0., 0., q[0], q[1], q[2], q[3]});
        const auto converged = Step(solver, beams);
        EXPECT_EQ(converged, true);
    }

    expect_kokkos_view_2D_equal(
        solver.state.q,
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
    std::array<double, 3> gravity = {0., 0., 0.};

    // Build vector of nodes (straight along x axis, no rotation)
    std::vector<BeamNode> beam_nodes;
    for (const double s : node_s) {
        beam_nodes.push_back(BeamNode(s, model.AddNode({10 * s + 2., 0., 0., 1., 0., 0., 0.})));
    }

    // Define beam initialization
    BeamsInput beams_input({BeamElement(beam_nodes, sections, quadrature)}, gravity);

    // Initialize beams from element inputs
    auto beams = CreateBeams(beams_input);

    // Add hub node and associated constraints
    float pitch = 0.;
    auto hub_node = model.AddNode({0., 0., 0., 1., 0., 0., 0.});
    model.AddRotationControl(hub_node, beam_nodes[0].node, {1., 0., 0.}, &pitch);
    model.AddFixedBC(hub_node);

    // Solution parameters
    const bool is_dynamic_solve(true);
    const int max_iter(5);
    const double step_size(0.01);  // seconds
    const double rho_inf(0.9);

    // Create solver
    Solver solver(
        is_dynamic_solve, max_iter, step_size, rho_inf, model.nodes, model.constraints, beams
    );

    // Perform 10 time steps and check for convergence within max_iter iterations
    for (int i = 0; i < 10; ++i) {
        double t = step_size * (i + 1);
        // Set pitch
        pitch = t * M_PI / 2.;
        const auto converged = Step(solver, beams);
        EXPECT_EQ(converged, true);
    }

    // Check that remaining displacements are as expected
    expect_kokkos_view_2D_equal(
        solver.state.q,
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

TEST(RotatingBeamTest, CylindricalConstraint) {
    auto model = Model();

    // Gravity vector
    std::array<double, 3> gravity = {0., 0., 0.};

    // Build vector of nodes (straight along x axis, no rotation)
    // Calculate displacement, velocity, acceleration assuming a
    // 0.1 rad/s angular velocity around the z axis
    const double omega = 0.1;
    std::vector<BeamNode> beam_nodes;
    for (const double s : node_s) {
        auto x = 10 * s + 2.;
        beam_nodes.push_back(BeamNode(
            s, model.AddNode(
                   {x, 0., 0., 1., 0., 0., 0.},        // position
                   {0., 0., 0., 1., 0., 0., 0.},       // displacement
                   {0., x * omega, 0., 0., 0., omega}  // velocity
               )
        ));
    }

    // Define beam initialization
    BeamsInput beams_input({BeamElement(beam_nodes, sections, quadrature)}, gravity);

    // Initialize beams from element inputs
    auto beams = CreateBeams(beams_input);

    // Add hub node and ground node
    auto hub_node = model.AddNode({0., 0., 0., 1., 0., 0., 0.});
    auto ground_node = model.AddNode({0, 0., -1., 1., 0., 0., 0.});

    // Add constraints
    model.AddFixedBC(ground_node);
    model.AddCylindricalConstraint(ground_node, hub_node);
    model.AddRigidConstraint(hub_node, beam_nodes[0].node);

    // Solution parameters
    const bool is_dynamic_solve(true);
    const int max_iter(5);
    const double step_size(0.01);  // seconds
    const double rho_inf(0.9);

    // Create solver
    Solver solver(
        is_dynamic_solve, max_iter, step_size, rho_inf, model.nodes, model.constraints, beams
    );

#ifdef OTURB_ENABLE_VTK
    UpdateState(beams, solver.state.q, solver.state.v, solver.state.vd);
    std::filesystem::remove_all("steps");
    std::filesystem::create_directory("steps");
    BeamsWriteVTK(beams, "steps/step_0000.vtu");
#endif

    // Run 10 steps
    for (int i = 0; i < 5; ++i) {
        const auto converged = Step(solver, beams);
        EXPECT_EQ(converged, true);
#ifdef OTURB_ENABLE_VTK
        auto tmp = std::to_string(i + 1);
        auto file_name = std::string("steps/step_") + std::string(4 - tmp.size(), '0') + tmp;
        BeamsWriteVTK(beams, file_name + ".vtu");
#endif
    }

    auto q = kokkos_view_2D_to_vector(solver.state.q);

    expect_kokkos_view_2D_equal(
        solver.state.q,
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

}  // namespace openturbine::restruct_poc::tests
