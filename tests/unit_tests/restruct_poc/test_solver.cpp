#include <initializer_list>

#include <gtest/gtest.h>

#include "test_utilities.hpp"

#include "src/restruct_poc/beams/beam_element.hpp"
#include "src/restruct_poc/beams/beam_node.hpp"
#include "src/restruct_poc/beams/beam_section.hpp"
#include "src/restruct_poc/beams/beams.hpp"
#include "src/restruct_poc/beams/beams_input.hpp"
#include "src/restruct_poc/beams/create_beams.hpp"
#include "src/restruct_poc/model/model.hpp"
#include "src/restruct_poc/solver/assemble_constraints.hpp"
#include "src/restruct_poc/solver/assemble_system.hpp"
#include "src/restruct_poc/solver/predict_next_state.hpp"
#include "src/restruct_poc/solver/solver.hpp"
#include "src/restruct_poc/solver/step.hpp"
#include "src/restruct_poc/types.hpp"

namespace openturbine::restruct_poc::tests {

class NewSolverTest : public testing::Test {
protected:
    // Per-test-suite set-up.
    // Called before the first test in this test suite.
    // Can be omitted if not needed.
    static void SetUpTestSuite() {
        // Mass matrix for uniform composite beam section
        std::array<std::array<double, 6>, 6> mass_matrix = {{
            {8.538e-2, 0., 0., 0., 0., 0.},
            {0., 8.538e-2, 0., 0., 0., 0.},
            {0., 0., 8.538e-2, 0., 0., 0.},
            {0., 0., 0., 1.4433e-2, 0., 0.},
            {0., 0., 0., 0., 0.40972e-2, 0.},
            {0., 0., 0., 0., 0., 1.0336e-2},
        }};

        // Stiffness matrix for uniform composite beam section
        std::array<std::array<double, 6>, 6> stiffness_matrix = {{
            {1368.17e3, 0., 0., 0., 0., 0.},
            {0., 88.56e3, 0., 0., 0., 0.},
            {0., 0., 38.78e3, 0., 0., 0.},
            {0., 0., 0., 16.9600e3, 17.6100e3, -0.3510e3},
            {0., 0., 0., 17.6100e3, 59.1200e3, -0.3700e3},
            {0., 0., 0., -0.3510e3, -0.3700e3, 141.470e3},
        }};

        // Create model for adding nodes and constraints
        auto model = Model_2();

        // Gravity vector
        std::array<double, 3> gravity = {0., 0., 0.};

        // Node locations (GLL quadrature)
        std::vector<double> node_s(
            {0., 0.11747233803526763, 0.35738424175967748, 0.64261575824032247, 0.88252766196473242,
             1.}
        );

        // Build vector of nodes (straight along x axis, no rotation)
        // Calculate displacement, velocity, acceleration assuming a
        // 0.1 rad/s angular velocity around the z axis
        const double omega = 0.1;
        std::vector<BeamNode> beam_nodes;
        for (const double s : node_s) {
            auto x = 10 * s + 2.;
            beam_nodes.push_back(BeamNode(
                s, *model.AddNode(
                       {x, 0., 0., 1., 0., 0., 0.},         // Position
                       {0., 0., 0., 1., 0., 0., 0.},        // Displacement
                       {0., x * omega, 0., 0., 0., omega},  // Velocity
                       {0., 0., 0., 0., 0., 0.}             // Acceleration
                   )
            ));
        }

        // Define beam initialization
        BeamsInput beams_input(
            {
                BeamElement(
                    beam_nodes,
                    {
                        BeamSection(0., mass_matrix, stiffness_matrix),
                        BeamSection(1., mass_matrix, stiffness_matrix),
                    },
                    BeamQuadrature{
                        {-0.9491079123427585, 0.1294849661688697},
                        {-0.7415311855993943, 0.27970539148927664},
                        {-0.40584515137739696, 0.3818300505051189},
                        {6.123233995736766e-17, 0.4179591836734694},
                        {0.4058451513773971, 0.3818300505051189},
                        {0.7415311855993945, 0.27970539148927664},
                        {0.9491079123427585, 0.1294849661688697},
                    }
                ),
            },
            gravity
        );

        // Initialize beams from element inputs
        beams_ = new Beams();
        *beams_ = CreateBeams(beams_input);

        // Constraint inputs
        model.AddPrescribedBC(*model.GetNode(0));

        // Solution parameters
        const bool is_dynamic_solve(true);
        const size_t max_iter(10);
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

        solver_ = new Solver(
            is_dynamic_solve, max_iter, step_size, rho_inf, nodes_vector, constraints_vector, *beams_
        );

        auto q = RotationVectorToQuaternion({0., 0., omega * step_size});
        solver_->constraints.UpdateDisplacement(0, {0., 0., 0., q[0], q[1], q[2], q[3]});

        // Predict the next state for the solver
        PredictNextState(*solver_);

        auto system_range = Kokkos::make_pair(0, solver_->num_system_dofs);
        auto constraint_range = Kokkos::make_pair(solver_->num_system_dofs, solver_->num_dofs);

        auto R_system = Kokkos::subview(solver_->R, system_range);
        auto R_lambda = Kokkos::subview(solver_->R, constraint_range);

        auto x_system = Kokkos::subview(solver_->x, system_range);
        auto x_lambda = Kokkos::subview(solver_->x, constraint_range);

        // Update beam elements state from solvers
        UpdateState(*beams_, solver_->state.q, solver_->state.v, solver_->state.vd);

        AssembleSystem(*solver_, *beams_, R_system);

        AssembleConstraints(*solver_, R_system, R_lambda);
    }

    // Per-test-suite tear-down.
    // Called after the last test in this test suite.
    // Can be omitted if not needed.
    static void TearDownTestSuite() {
        delete beams_;
        beams_ = nullptr;
        delete solver_;
        solver_ = nullptr;
    }

    // Some expensive resource shared by all tests.
    static Beams* beams_;
    static Solver* solver_;
};

Beams* NewSolverTest::beams_ = nullptr;
Solver* NewSolverTest::solver_ = nullptr;

TEST_F(NewSolverTest, SolverPredictNextState_lambda) {
    expect_kokkos_view_1D_equal(solver_->state.lambda, {0., 0., 0., 0., 0., 0.});
}

TEST_F(NewSolverTest, SolverPredictNextState_q_prev) {
    expect_kokkos_view_2D_equal(
        solver_->state.q_prev,
        {
            {0., 0., 0., 1., 0., 0., 0.},
            {0., 0., 0., 1., 0., 0., 0.},
            {0., 0., 0., 1., 0., 0., 0.},
            {0., 0., 0., 1., 0., 0., 0.},
            {0., 0., 0., 1., 0., 0., 0.},
            {0., 0., 0., 1., 0., 0., 0.},
        }
    );
}

TEST_F(NewSolverTest, SolverPredictNextState_a) {
    expect_kokkos_view_2D_equal(
        solver_->state.a,
        {
            {0., 0., 0., 0., 0., 0.},
            {0., 0., 0., 0., 0., 0.},
            {0., 0., 0., 0., 0., 0.},
            {0., 0., 0., 0., 0., 0.},
            {0., 0., 0., 0., 0., 0.},
            {0., 0., 0., 0., 0., 0.},
        }
    );
}

TEST_F(NewSolverTest, SolverPredictNextState_v) {
    expect_kokkos_view_2D_equal(
        solver_->state.v,
        {
            {0, 0.20000000000000000, 0, 0, 0, 0.1},
            {0, 0.31747233803526764, 0, 0, 0, 0.1},
            {0, 0.55738424175967749, 0, 0, 0, 0.1},
            {0, 0.84261575824032242, 0, 0, 0, 0.1},
            {0, 1.08252766196473240, 0, 0, 0, 0.1},
            {0, 1.20000000000000000, 0, 0, 0, 0.1},
        }
    );
}

TEST_F(NewSolverTest, SolverPredictNextState_vd) {
    expect_kokkos_view_2D_equal(
        solver_->state.vd,
        {
            {0., 0., 0., 0., 0., 0.},
            {0., 0., 0., 0., 0., 0.},
            {0., 0., 0., 0., 0., 0.},
            {0., 0., 0., 0., 0., 0.},
            {0., 0., 0., 0., 0., 0.},
            {0., 0., 0., 0., 0., 0.},
        }
    );
}

TEST_F(NewSolverTest, SolverPredictNextState_q_delta) {
    expect_kokkos_view_2D_equal(
        solver_->state.q_delta,
        {
            {0., 0.20000000000000001, 0., 0., 0., 0.10000000000000001},
            {0., 0.31747233803526764, 0., 0., 0., 0.10000000000000001},
            {0., 0.55738424175967749, 0., 0., 0., 0.10000000000000001},
            {0., 0.84261575824032242, 0., 0., 0., 0.10000000000000001},
            {0., 1.08252766196473240, 0., 0., 0., 0.10000000000000001},
            {0., 1.20000000000000020, 0., 0., 0., 0.10000000000000001},
        }
    );
}

TEST_F(NewSolverTest, SolverPredictNextState_q) {
    expect_kokkos_view_2D_equal(
        solver_->state.q,
        {
            {0, 0.0020000000000000, 0, 0.999999875, 0, 0, 0.0004999999},
            {0, 0.0031747233803526, 0, 0.999999875, 0, 0, 0.0004999999},
            {0, 0.0055738424175967, 0, 0.999999875, 0, 0, 0.0004999999},
            {0, 0.0084261575824032, 0, 0.999999875, 0, 0, 0.0004999999},
            {0, 0.0108252766196473, 0, 0.999999875, 0, 0, 0.0004999999},
            {0, 0.0120000000000000, 0, 0.999999875, 0, 0, 0.0004999999},
        }
    );
}

TEST_F(NewSolverTest, ConstraintResidualVector) {
    expect_kokkos_view_1D_equal(
        solver_->constraints.Phi,
        {
            9.9999991642896191E-7,
            3.3333331667800836E-10,
            0.,
            0.,
            0.,
            0.,
        }
    );
}

TEST_F(NewSolverTest, ConstraintGradientMatrix) {
    expect_kokkos_view_2D_equal(
        Kokkos::subview(solver_->constraints.B, Kokkos::make_pair(0, 6), Kokkos::make_pair(0, 6)),
        {
            {1., 0., 0., 0.0000000000000000, 0.0000000000000000, 0.0000000000000000},
            {0., 1., 0., 0.0000000000000000, 0.0000000000000000, 0.0000000000000000},
            {0., 0., 1., 0.0000000000000000, 0.0000000000000000, 0.0000000000000000},
            {0., 0., 0., 1.0000000000000004, 0.0000000000000000, 0.0000000000000000},
            {0., 0., 0., 0.0000000000000000, 1.0000000000000004, 0.0000000000000000},
            {0., 0., 0., 0.0000000000000000, 0.0000000000000000, 1.0000000000000004},
        }
    );
}

TEST_F(NewSolverTest, AssembleResidualVector) {
    expect_kokkos_view_1D_equal(
        solver_->R,
        {
            -0.68408451644565105,
            -0.00065456473269251652,
            0,
            1.4663215017154278E-17,
            1.5487939846687054E-17,
            0.0000098399278916272914,
            -4.9960036108132044E-16,
            7.2063733561229804E-15,
            0,
            -9.2033949200707357E-18,
            -9.7210350349908583E-18,
            0.000055862494393220151,
            -8.1878948066105295E-16,
            -3.4811400723838704E-14,
            0,
            -4.27839293851558E-18,
            -4.5190289029179014E-18,
            0.000081896496420578814,
            1.7208456881689926E-15,
            1.0463827612543219E-13,
            0,
            1.991039683802277E-17,
            2.1030246653970917E-17,
            0.000081896496635873786,
            -5.1070259132757201E-15,
            -3.455288355783126E-13,
            0,
            -7.9977507519330913E-17,
            -8.4475800436550718E-17,
            0.000055862494238845577,
            0.68408451644565516,
            0.00065456473296101171,
            0,
            5.8885683522740175E-17,
            6.2197677873801522E-17,
            0.0000098399277591661938,
            9.9999991642896191E-7,
            3.3333331667800836E-10,
            0,
            0,
            0,
            0,
        }
    );
}

TEST_F(NewSolverTest, DISABLED_AssembleIterationMatrix) {
    expect_kokkos_view_2D_equal(
        Kokkos::subview(solver_->St, Kokkos::make_pair(0, 12), Kokkos::make_pair(0, 12)),
        {
            {1414801.7504034417, 1322.2627851577333, 0.0, 0.0, 0.0, -44.2794594494287,
             -1574613.1137289538, -1472.9196463832786, 0.0, 0.0, 0.0, -59.87418876160718},
            {1322.2627851577329, 92540.72826322596, 0.0, 0.0, 0.0, 44279.65795795503,
             -1472.9196463832784, -101695.43123906071, 0.0, 0.0, 0.0, 59874.45718261191},
            {0.0, 0.0, 41100.0726666667, 29.08449765683834, -19389.645031296728, 0.0, 0.0, 0.0,
             -44393.6958340446, 39.327731738255004, -26218.460682663215, 0.0},
            {0.0, 0.0, 19.389672717633655, 17653.597868705176, 18150.55055727363, -362.3174853804042,
             0.0, 0.0, -26.218498119817678, -19430.15450862125, -20234.27111826118,
             403.5994572601714},
            {0.0, 0.0, -19389.657957741776, 18105.353944134222, 72936.68675438849,
             -382.6958421062328, 0.0, 0.0, 26218.478161657324, -20191.88748468507,
             -65290.151252854055, 426.29969681875457},
            {-44.27945944942881, 44279.65795795503, 0.0, -362.1260770890552, -382.8769370511865,
             173146.22637419065, 59.8741887616077, -59874.45718261189, 0.0, 403.386240162955,
             426.50142548061893, -156418.0506742802},
            {-1574613.1137289538, -1472.9196463832789, 0.0, 0.0, 0.0, 59.874188761607556,
             2503202.5925416285, 2335.7187569544285, 0.0, 0.0, 0.0, 3.907985046680551e-14},
            {-1472.9196463832784, -101695.43123906071, 0.0, 0.0, 0.0, -59874.4571826119,
             2335.7187569544276, 167486.94987970704, 0.0, 0.0, 0.0, 1.921307557495311e-11},
            {0.0, 0.0, -44393.69583404459, -39.32773173825499, 26218.460682663215, 0.0, 0.0, 0.0,
             76619.30579594545, -1.1182165464503461e-15, 2.4584775114948027e-12, 0.0},
            {0.0, 0.0, 26.218498119817696, -19430.15450862125, -20234.27111826118, 403.5994572601714,
             0.0, 0.0, -1.887379141862766e-15, 31863.56207519658, 32016.823623811943,
             -640.0178210223318},
            {0.0, 0.0, -26218.478161657324, -20191.887484685067, -65290.15125285405,
             426.29969681875457, 0.0, 0.0, 2.4584778657299466e-12, 31913.957377672083,
             174987.68054334674, -676.0152873162565},
            {-59.87418876160725, 59874.45718261191, 0.0, 403.386240162955, 426.50142548061893,
             -156418.0506742802, 1.212363542890671e-13, 1.6086687537608668e-11, 0.0,
             -639.6797067372147, -676.3351835308863, 411288.96761996194},
        }
    );
}

class SolverStep1Test : public testing::Test {
protected:
    // Per-test-suite set-up.
    // Called before the first test in this test suite.
    // Can be omitted if not needed.
    static void SetUpTestSuite() {
        // Mass matrix for uniform composite beam section
        std::array<std::array<double, 6>, 6> mass_matrix = {{
            {8.538e-2, 0., 0., 0., 0., 0.},
            {0., 8.538e-2, 0., 0., 0., 0.},
            {0., 0., 8.538e-2, 0., 0., 0.},
            {0., 0., 0., 1.4433e-2, 0., 0.},
            {0., 0., 0., 0., 0.40972e-2, 0.},
            {0., 0., 0., 0., 0., 1.0336e-2},
        }};

        // Stiffness matrix for uniform composite beam section
        std::array<std::array<double, 6>, 6> stiffness_matrix = {{
            {1368.17e3, 0., 0., 0., 0., 0.},
            {0., 88.56e3, 0., 0., 0., 0.},
            {0., 0., 38.78e3, 0., 0., 0.},
            {0., 0., 0., 16.9600e3, 17.6100e3, -0.3510e3},
            {0., 0., 0., 17.6100e3, 59.1200e3, -0.3700e3},
            {0., 0., 0., -0.3510e3, -0.3700e3, 141.470e3},
        }};

        // Create model for adding nodes and constraints
        auto model = Model_2();

        // Gravity vector
        std::array<double, 3> gravity = {0., 0., 0.};

        // Node locations (GLL quadrature)
        std::vector<double> node_s({
            0.,
            0.11747233803526763,
            0.35738424175967748,
            0.64261575824032247,
            0.88252766196473242,
            1.,
        });

        // Build vector of nodes (straight along x axis, no rotation)
        // Calculate displacement, velocity, acceleration assuming a
        // 0.1 rad/s angular velocity around the z axis
        const double omega = 0.1;
        std::vector<BeamNode> beam_nodes;
        for (const double s : node_s) {
            auto x = 10 * s + 2.;
            beam_nodes.push_back(BeamNode(
                s, *model.AddNode(
                       {x, 0., 0., 1., 0., 0., 0.},         // Position
                       {0., 0., 0., 1., 0., 0., 0.},        // Displacement
                       {0., x * omega, 0., 0., 0., omega},  // Velocity
                       {0., 0., 0., 0., 0., 0.}             // Acceleration
                   )
            ));
        }

        // Define beam initialization
        BeamsInput beams_input(
            {
                BeamElement(
                    beam_nodes,
                    {
                        BeamSection(0., mass_matrix, stiffness_matrix),
                        BeamSection(1., mass_matrix, stiffness_matrix),
                    },
                    BeamQuadrature{
                        {-0.9491079123427585, 0.1294849661688697},
                        {-0.7415311855993943, 0.27970539148927664},
                        {-0.40584515137739696, 0.3818300505051189},
                        {6.123233995736766e-17, 0.4179591836734694},
                        {0.4058451513773971, 0.3818300505051189},
                        {0.7415311855993945, 0.27970539148927664},
                        {0.9491079123427585, 0.1294849661688697},
                    }
                ),
            },
            gravity
        );

        // Initialize beams from element inputs
        beams_ = new Beams();
        *beams_ = CreateBeams(beams_input);

        // Constraint inputs
        model.AddPrescribedBC(*model.GetNode(0));

        // Solution parameters
        const bool is_dynamic_solve(true);
        const size_t max_iter(0);
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
        solver_ = new Solver(
            is_dynamic_solve, max_iter, step_size, rho_inf, nodes_vector, constraints_vector, *beams_
        );

        // Set constraint displacement
        auto q = RotationVectorToQuaternion({0., 0., omega * step_size});
        solver_->constraints.UpdateDisplacement(0., {0., 0., 0., q[0], q[1], q[2], q[3]});

        // Perform step with 1 convergence iteration
        Step(*solver_, *beams_);
    }

    // Per-test-suite tear-down.
    // Called after the last test in this test suite.
    // Can be omitted if not needed.
    static void TearDownTestSuite() {
        delete beams_;
        beams_ = nullptr;
        delete solver_;
        solver_ = nullptr;
    }

    // Some expensive resource shared by all tests.
    static Beams* beams_;
    static Solver* solver_;
};

Beams* SolverStep1Test::beams_ = nullptr;
Solver* SolverStep1Test::solver_ = nullptr;

TEST_F(SolverStep1Test, SolutionVector) {
    expect_kokkos_view_1D_equal(
        solver_->x,
        {
            -9.9999991642894645E-7,    -3.3333331667800779E-10, -2.7693137528041648E-28,
            -6.5954208494030506E-30,   1.3190840599775882E-26,  7.7898219222917266E-24,
            -0.0000014964806473087194, -7.295038225205836E-10,  -4.5434716051775647E-14,
            -3.1920684921905924E-12,   -4.0723290197473225E-14, -2.2269841387499837E-10,
            -0.0000025281897479172782, -1.9662810249790896E-9,  -3.0976187800764434E-14,
            -4.8228594588737896E-12,   -9.9091066510552565E-14, -4.1568756453239842E-10,
            -0.0000038030054024772931, -3.5935658864740807E-9,  6.4411049537235008E-14,
            -3.6024743778286134E-12,   -1.1579932606573532E-13, -4.3409555957226201E-10,
            -0.000004933638516780269,  -4.9475925730408304E-9,  2.5185545987603492E-13,
            -2.7436857930422583E-12,   -1.6107798285714521E-13, -4.2116615327927008E-10,
            -0.0000055119072875186336, -5.625802843908016E-9,   4.2749628020387186E-13,
            -2.6313864405287534E-12,   -1.7915982090983506E-13, -4.1964251580055361E-10,
            0.10816660597819647,       0.000095455157310304377, -2.1220160418770112E-9,
            2.0316113427113777E-8,     2.1788291524533811E-8,   -0.000033726153743119725,
        }
    );
}

TEST_F(SolverStep1Test, ConvergenceError) {
    EXPECT_NEAR(solver_->convergence_err[0], 14.796392074134879, 1.e-7);
}

TEST_F(SolverStep1Test, SolverUpdateStatePrediction_q_delta) {
    expect_kokkos_view_2D_equal(
        solver_->state.q_delta,
        {
            {-0.000099999991642894646, 0.19999996666666833, -2.7693137528041646E-26,
             -6.5954208494030504E-28, 1.3190840599775882E-24, 0.10000000000000001},
            {-0.00014964806473087192, 0.31747226508488541, -4.5434716051775645E-12,
             -3.1920684921905925E-10, -4.072329019747322E-12, 0.099999977730158617},
            {-0.00025281897479172784, 0.55738404513157502, -3.0976187800764433E-12,
             -4.8228594588737892E-10, -9.9091066510552555E-12, 0.099999958431243555},
            {-0.00038030054024772928, 0.84261539888373382, 6.4411049537235007E-12,
             -3.6024743778286132E-10, -1.1579932606573532E-11, 0.099999956590444042},
            {-0.00049336385167802685, 1.082527167205475, 2.5185545987603493E-11,
             -2.743685793042258E-10, -1.6107798285714522E-11, 0.099999957883384671},
            {-0.00055119072875186334, 1.1999994374197158, 4.2749628020387184E-11,
             -2.6313864405287535E-10, -1.7915982090983507E-11, 0.099999958035748431},
        }
    );
}

TEST_F(SolverStep1Test, SolverUpdateStatePrediction_v) {
    expect_kokkos_view_2D_equal(
        solver_->state.v,
        {
            {-0.00019949998332757481, 0.19999993350000334, -5.5247809368443091E-26,
             -1.3157864594559085E-27, 2.6315726996552886E-24, 0.10000000000000001},
            {-0.00029854788913808952, 0.31747219249925507, -9.0642258523292409E-12,
             -6.3681766419202313E-10, -8.1242963943959083E-12, 0.099999955571666437},
            {-0.00050437385470949699, 0.55738384948661301, -6.1797494662525043E-12,
             -9.62160462045321E-10, -1.9768667768855238E-11, 0.099999917070330888},
            {-0.00075869957779421996, 0.84261504132392806, 1.2850004382678384E-11,
             -7.186936383768084E-10, -2.3101965550114195E-11, 0.099999913397935874},
            {-0.00098426088409766361, 1.082526674920014, 5.0245164245268966E-11,
             -5.4736531571193053E-10, -3.2135057580000468E-11, 0.09999991597735243},
            {-0.0010996255038599673, 1.1999988776523327, 8.5285507900672438E-11,
             -5.2496159488548634E-10, -3.5742384271512094E-11, 0.099999916281318101},
        }
    );
}

TEST_F(SolverStep1Test, SolverUpdateStatePrediction_vd) {
    expect_kokkos_view_2D_equal(
        solver_->state.vd,
        {
            {-0.039709996681393474, -0.000013236666005283692, -1.0996944912385341E-23,
             -2.619041619297952E-25, 5.2380828021710043E-22, 3.0933382853420454E-19},
            {-0.059425246504629256, -0.000028968596792292381, -1.8042125744160112E-9,
             -1.2675703982488845E-7, -1.617121853741662E-9, -0.0000088433540149761862},
            {-0.10039441488979514, -0.000078081019501919665, -1.2300644175683558E-9,
             -1.9151574911187822E-7, -3.9349062511340432E-9, -0.000016506953187581543},
            {-0.15101734453237334, -0.00014270050135188576, 2.5577627771236027E-9,
             -1.4305425754357427E-7, -4.5983912380703506E-9, -0.000017237934670614527},
            {-0.19591478550134453, -0.00019646890107545141, 1.0001180311677349E-8,
             -1.0895176284170809E-7, -6.3964066992572377E-9, -0.000016724507946719819},
            {-0.21887783838736499, -0.00022340063093158737, 1.6975877286895753E-8,
             -1.0449235555339682E-7, -7.1144364883295513E-9, -0.000016664004302439988},
        }
    );
}

TEST_F(SolverStep1Test, SolverUpdateStatePrediction_q) {
    expect_kokkos_view_2D_equal(
        solver_->state.q,
        {
            {-9.9999991642894645E-7, 0.0019999996666666834, -2.7693137528041648E-28,
             0.99999987500000264, -3.2977102872969255E-30, 6.5954200250787654E-27,
             0.0004999999791666669},
            {-0.0000014964806473087194, 0.003174722650848854, -4.5434716051775647E-14,
             0.99999987500005827, -1.5960341795939002E-12, -2.0361644250335121E-14,
             0.00049999986781747402},
            {-0.0000025281897479172786, 0.0055738404513157504, -3.0976187800764434E-14,
             0.99999987500010656, -2.4114296289607408E-12, -4.9545531190880803E-14,
             0.0004999997713229107},
            {-0.0000038030054024772931, 0.0084261539888373388, 6.4411049537235008E-14,
             0.99999987500011112, -1.8012371138628233E-12, -5.7899660620383827E-14,
             0.0004999997621189143},
            {-0.000004933638516780269, 0.010825271672054751, 2.5185545987603492E-13,
             0.99999987500010789, -1.3718428393610572E-12, -8.0538988072784165E-14,
             0.00049999976858361656},
            {-0.0000055119072875186336, 0.011999994374197159, 4.2749628020387186E-13,
             0.99999987500010756, -1.3156931654438724E-12, -8.9579906722424441E-14,
             0.00049999976934543534},
        }
    );
}

TEST_F(SolverStep1Test, SolverUpdateStatePrediction_lambda) {
    expect_kokkos_view_1D_equal(
        solver_->state.lambda,
        {
            -0.10816660597819647,
            -0.000095455157310304377,
            2.1220160418770112E-9,
            -2.0316113427113777E-8,
            -2.1788291524533811E-8,
            0.000033726153743119725,
        }
    );
}

class SolverStep2Test : public testing::Test {
protected:
    // Per-test-suite set-up.
    // Called before the first test in this test suite.
    // Can be omitted if not needed.
    static void SetUpTestSuite() {
        // Mass matrix for uniform composite beam section
        std::array<std::array<double, 6>, 6> mass_matrix = {{
            {8.538e-2, 0., 0., 0., 0., 0.},
            {0., 8.538e-2, 0., 0., 0., 0.},
            {0., 0., 8.538e-2, 0., 0., 0.},
            {0., 0., 0., 1.4433e-2, 0., 0.},
            {0., 0., 0., 0., 0.40972e-2, 0.},
            {0., 0., 0., 0., 0., 1.0336e-2},
        }};

        // Stiffness matrix for uniform composite beam section
        std::array<std::array<double, 6>, 6> stiffness_matrix = {{
            {1368.17e3, 0., 0., 0., 0., 0.},
            {0., 88.56e3, 0., 0., 0., 0.},
            {0., 0., 38.78e3, 0., 0., 0.},
            {0., 0., 0., 16.9600e3, 17.6100e3, -0.3510e3},
            {0., 0., 0., 17.6100e3, 59.1200e3, -0.3700e3},
            {0., 0., 0., -0.3510e3, -0.3700e3, 141.470e3},
        }};

        // Create model for adding nodes and constraints
        auto model = Model_2();

        // Gravity vector
        std::array<double, 3> gravity = {0., 0., 0.};

        // Node locations (GLL quadrature)
        std::vector<double> node_s(
            {0., 0.11747233803526763, 0.35738424175967748, 0.64261575824032247, 0.88252766196473242,
             1.}
        );

        // Build vector of nodes (straight along x axis, no rotation)
        // Calculate displacement, velocity, acceleration assuming a
        // 0.1 rad/s angular velocity around the z axis
        const double omega = 0.1;
        std::vector<BeamNode> beam_nodes;
        for (const double s : node_s) {
            auto x = 10 * s + 2.;
            beam_nodes.push_back(BeamNode(
                s, *model.AddNode(
                       {x, 0., 0., 1., 0., 0., 0.},         // Position
                       {0., 0., 0., 1., 0., 0., 0.},        // Displacement
                       {0., x * omega, 0., 0., 0., omega},  // Velocity
                       {0., 0., 0., 0., 0., 0.}             // Acceleration
                   )
            ));
        }

        // Define beam initialization
        BeamsInput beams_input(
            {
                BeamElement(
                    beam_nodes,
                    {
                        BeamSection(0., mass_matrix, stiffness_matrix),
                        BeamSection(1., mass_matrix, stiffness_matrix),
                    },
                    BeamQuadrature{
                        {-0.9491079123427585, 0.1294849661688697},
                        {-0.7415311855993943, 0.27970539148927664},
                        {-0.40584515137739696, 0.3818300505051189},
                        {6.123233995736766e-17, 0.4179591836734694},
                        {0.4058451513773971, 0.3818300505051189},
                        {0.7415311855993945, 0.27970539148927664},
                        {0.9491079123427585, 0.1294849661688697},
                    }
                ),
            },
            gravity
        );

        // Initialize beams from element inputs
        beams_ = new Beams();
        *beams_ = CreateBeams(beams_input);

        // Constraint inputs
        model.AddPrescribedBC(*model.GetNode(0));

        // Solution parameters
        const bool is_dynamic_solve(true);
        const size_t max_iter(2);
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
        solver_ = new Solver(
            is_dynamic_solve, max_iter, step_size, rho_inf, nodes_vector, constraints_vector, *beams_
        );

        // Set constraint displacement
        auto q = RotationVectorToQuaternion({0., 0., omega * step_size});
        solver_->constraints.UpdateDisplacement(0., {0., 0., 0., q[0], q[1], q[2], q[3]});

        // Perform step with 1 convergence iteration
        Step(*solver_, *beams_);
    }

    // Per-test-suite tear-down.
    // Called after the last test in this test suite.
    // Can be omitted if not needed.
    static void TearDownTestSuite() {
        delete beams_;
        beams_ = nullptr;
        delete solver_;
        solver_ = nullptr;
    }

    // Some expensive resource shared by all tests.
    static Beams* beams_;
    static Solver* solver_;
};

Beams* SolverStep2Test::beams_ = nullptr;
Solver* SolverStep2Test::solver_ = nullptr;

TEST_F(SolverStep2Test, ConstraintResidualVector) {
    expect_kokkos_view_1D_equal(
        solver_->constraints.Phi,
        {
            0.0,
            0.0,
            -2.769313752804165e-28,
            -1.31908395004359e-29,
            1.3190835103592412e-26,
            0.0,
        }
    );
}

TEST_F(SolverStep2Test, ConstraintGradientMatrix) {
    expect_kokkos_view_2D_equal(
        Kokkos::subview(solver_->constraints.B, Kokkos::make_pair(0, 6), Kokkos::make_pair(0, 6)),
        {
            {1.0, 0.0, 0.0, 0.0, 0.0, 0.0},
            {0.0, 1.0, 0.0, 0.0, 0.0, 0.0},
            {0.0, 0.0, 1.0, 0.0, 0.0, 0.0},
            {0.0, 0.0, 0.0, 1.0000000000000004, 0.0, -6.595417551796206e-27},
            {0.0, 0.0, 0.0, 0.0, 1.0000000000000004, -6.59541975021795e-30},
            {0.0, 0.0, 0.0, 6.595417551796206e-27, 6.595419750217949e-30, 1.0000000000000002},
        }
    );
}

TEST_F(SolverStep2Test, ResidualVector) {
    expect_kokkos_view_1D_equal(
        solver_->R,
        {
            -0.0000059926097477785435,
            -5.288375554255456E-9,
            1.1889571903333178E-13,
            -1.1240653642481715E-12,
            -1.2020880297284837E-12,
            1.8684840013084998E-9,
            -3.6875789625611664E-15,
            -4.1819420587222298E-15,
            3.7352345516767378E-15,
            -1.1877649082466982E-15,
            -1.0506791059026386E-14,
            -5.8939891699170042E-15,
            -2.8792767262448265E-16,
            -1.6256730932454705E-15,
            -3.7325943099788989E-16,
            -5.3808016138201058E-16,
            -1.6282824212620447E-14,
            -1.1016714995625436E-14,
            5.2084529364118357E-16,
            -1.1638779207501438E-16,
            -1.4881053281336617E-15,
            9.5313284912395902E-17,
            -1.0413534631705933E-14,
            -1.3446531440415119E-14,
            2.7455938415381741E-15,
            -2.5729773964724179E-16,
            -3.7154086685782557E-16,
            1.5306103158449729E-16,
            -5.1065389766195233E-15,
            -9.9765062644864507E-15,
            -9.9677609771866959E-16,
            7.3693341584230349E-15,
            -2.8348323425684356E-15,
            3.4513177152054705E-17,
            -8.3796234226909556E-16,
            -1.8000463717889823E-15,
            0,
            0,
            -2.7693137528041648E-28,
            -1.3190839500435901E-29,
            1.3190835103592412E-26,
            0,
        }
    );
}

TEST_F(SolverStep2Test, DISABLED_IterationMatrix) {
    expect_kokkos_view_2D_equal(
        Kokkos::subview(solver_->St, Kokkos::make_pair(0, 12), Kokkos::make_pair(0, 12)),
        {
            {39.19118422170844, 0.03662777474795655, 5.431970858957439e-13, 1.6028381978290388e-15,
             -1.0886758979566943e-12, -0.0012265887214511358, -43.6180917930529,
             -0.04080109468702067, -6.645402921711687e-13, 2.1336648365260163e-15,
             -1.4133438319901642e-12, -0.0016585796701074095},
            {0.03662777474795655, 2.563455076537186, -1.812430399129996e-12, 2.92254394582245e-14,
             6.342242601248811e-13, 1.2265907994587366, -0.04080109468702067, -2.817047956753582,
             1.8714070969396966e-12, -2.0489986853981713e-14, 4.561344703178226e-12,
             1.6585823496633025},
            {5.431970858957439e-13, -1.812430399129996e-12, 1.1385061680517088,
             0.0008056762368014134, -0.5371170480229618, -6.352680946684192e-13,
             -6.645402921711687e-13, 1.871407096939697e-12, -1.229742266871041,
             0.0010894262602801842, -0.7262836716663055, -7.563947523418205e-13},
            {1.0585001152893736e-15, 2.9542549152820547e-14, 0.0005371176456544514,
             0.4890193316384446, 0.5027853299013584, -0.010036495441208484, -1.4575395534518318e-15,
             -2.889750631789036e-14, -0.000726284456581465, -0.5382314269900914, -0.5605061258067667,
             0.011180040370362227},
            {-1.0886766086528004e-12, 6.342095945531298e-13, -0.5371174061013255, 0.5015333431792913,
             2.0204114163031464, -0.01060099285205352, 1.4876296152371372e-12,
             -1.192860296608879e-12, 0.7262841601176612, -0.559332065668402, -1.8085904088500921,
             0.011808855865582644},
            {-0.0012265887214511358, 1.2265907994587366, -6.35268094668419e-13, -0.01003119327307262,
             -0.010606009331956042, 4.7962985870997965, 0.001658579648880224, -1.6585823537657316,
             1.1943178361623309e-12, 0.011174134079342342, 0.011814443906856665, -4.332909032644746},
            {-43.618091793052905, -0.040801094687020666, -6.645402921711687e-13,
             -2.201354056150927e-15, 1.4876286385291646e-12, 0.001658579648880224, 69.34079203718417,
             0.06470134063641056, 2.1638441155967127e-12, 1.8945497940296173e-16,
             -8.66663142345895e-14, -7.04516117205331e-10},
            {-0.040801094687020666, -2.817047956753582, 1.8714070969396966e-12,
             -2.8301071403057902e-14, -1.1928745465507937e-12, -1.6585823537657314,
             0.06470134063641056, 4.639527697478868, -4.9325999265727314e-12, -6.226200383309214e-14,
             1.8708422050523935e-12, 1.2558433637842626e-7},
            {-6.645402921711687e-13, 1.8714070969396974e-12, -1.229742266871041,
             -0.0010894263853253655, 0.7262836759281092, 1.1943178361623313e-12,
             2.163844115596713e-12, -4.932599926572733e-12, 2.122418443100981, 4.322686056153791e-11,
             -1.3047062064454673e-7, -1.8710188932725623e-12},
            {1.3936626810880922e-15, 1.5120959585306213e-14, 0.0007262844145473438,
             -0.5382314269271556, -0.5605061268801798, 0.011180040368717822, 1.4612180199203463e-16,
             -6.13265752267249e-14, -2.2008438829458875e-11, 0.8826471493082577, 0.8868925949620642,
             -0.01772902551933346},
            {-1.4106976254619134e-12, 1.914194709556308e-12, -0.7262841558557008,
             -0.5593320644536077, -1.8085904089118592, 0.011808855863447563, -8.66664006154342e-14,
             1.870872979706598e-12, -1.3047063138542513e-7, 0.8840431033768114, 4.847330412009456,
             -0.01872618523302892},
            {-0.0016585796701074067, 1.6585823496633025, -1.9155883722373963e-12,
             0.011174134078697226, 0.011814443914277478, -4.332909032644728, -7.045161177250753e-10,
             1.2558433647290286e-7, -1.8710191015085898e-12, -0.017719659473594083,
             -0.018735046596094138, 11.393070412332664},
        }
    );
}

TEST_F(SolverStep2Test, SolutionVector) {
    expect_kokkos_view_1D_equal(
        solver_->x,
        {
            2.0701786547467534E-22,  -9.7804349489591981E-26, 2.7272832797488856E-28,
            2.0547315819961114E-29,  -1.4712947016254835E-26, 5.4331648183819847E-25,
            3.8348324508389868E-17,  9.4969920440321861E-16,  -1.3580769206195316E-17,
            -5.2335075093141956E-18, 3.1855954047525528E-15,  9.2846344878465848E-16,
            -6.0860076883771766E-17, 8.028169690742347E-16,   -1.479979763477676E-17,
            -2.7808660671715477E-17, 4.8007090441525481E-15,  8.6051705604834966E-16,
            -2.179748742218857E-16,  -2.7745453151127729E-15, 1.1652103769994212E-17,
            -3.0548898515545171E-17, 3.569536108741796E-15,   -4.1743489803882303E-17,
            -3.0721911895997357E-16, -9.1411516388420519E-15, 7.8855513035796911E-17,
            -6.4825821213015845E-17, 2.6903185401857071E-15,  -1.1004628463528681E-15,
            -2.8438123968141702E-16, -1.3829871537755664E-14, 1.4293886089416713E-16,
            -7.9353195214248362E-17, 2.5707926448736107E-15,  -1.3780235858853489E-15,
            0.21633321196105559,     0.00019091040135282763,  -4.2445828198259645E-9,
            4.0632386057053241E-8,   4.3576751058673394E-8,   -0.000067452108435714744,
        }
    );
}

// TEST_F(SolverStep2Test, ConvergenceError) {
//     EXPECT_NEAR(solver_->convergence_err, 13.698797186349344, 1.e-9);
// }

// TEST_F(SolverStep2Test, SolverUpdateStatePrediction_q_delta) {
//     expect_kokkos_view_2D_equal(
//         solver_->state.q_delta,
//         {
//             {-0.000099999991642894646, 0.19999996666666833, -2.7693137528041646E-26,
//              -6.5954208494030504E-28, 1.3190840599775882E-24, 0.10000000000000001},
//             {-0.00014964806473087192, 0.31747226508488541, -4.5434716051775645E-12,
//              -3.1920684921905925E-10, -4.072329019747322E-12, 0.099999977730158617},
//             {-0.00025281897479172784, 0.55738404513157502, -3.0976187800764433E-12,
//              -4.8228594588737892E-10, -9.9091066510552555E-12, 0.099999958431243555},
//             {-0.00038030054024772928, 0.84261539888373382, 6.4411049537235007E-12,
//              -3.6024743778286132E-10, -1.1579932606573532E-11, 0.099999956590444042},
//             {-0.00049336385167802685, 1.082527167205475, 2.5185545987603493E-11,
//              -2.743685793042258E-10, -1.6107798285714522E-11, 0.099999957883384671},
//             {-0.00055119072875186334, 1.1999994374197158, 4.2749628020387184E-11,
//              -2.6313864405287535E-10, -1.7915982090983507E-11, 0.099999958035748431},
//         }
//     );
// }

// TEST_F(SolverStep2Test, SolverUpdateStatePrediction_v) {
//     expect_kokkos_view_2D_equal(
//         solver_->state.v,
//         {
//             {-0.00019949998332757481, 0.19999993350000334, -5.5247809368443091E-26,
//              -1.3157864594559085E-27, 2.6315726996552886E-24, 0.10000000000000001},
//             {-0.00029854788913808952, 0.31747219249925507, -9.0642258523292409E-12,
//              -6.3681766419202313E-10, -8.1242963943959083E-12, 0.099999955571666437},
//             {-0.00050437385470949699, 0.55738384948661301, -6.1797494662525043E-12,
//              -9.62160462045321E-10, -1.9768667768855238E-11, 0.099999917070330888},
//             {-0.00075869957779421996, 0.84261504132392806, 1.2850004382678384E-11,
//              -7.186936383768084E-10, -2.3101965550114195E-11, 0.099999913397935874},
//             {-0.00098426088409766361, 1.082526674920014, 5.0245164245268966E-11,
//              -5.4736531571193053E-10, -3.2135057580000468E-11, 0.09999991597735243},
//             {-0.0010996255038599673, 1.1999988776523327, 8.5285507900672438E-11,
//              -5.2496159488548634E-10, -3.5742384271512094E-11, 0.099999916281318101},
//         }
//     );
// }

// TEST_F(SolverStep2Test, SolverUpdateStatePrediction_vd) {
//     expect_kokkos_view_2D_equal(
//         solver_->state.vd,
//         {
//             {-0.039709996681393474, -0.000013236666005283692, -1.0996944912385341E-23,
//              -2.619041619297952E-25, 5.2380828021710043E-22, 3.0933382853420454E-19},
//             {-0.059425246504629256, -0.000028968596792292381, -1.8042125744160112E-9,
//              -1.2675703982488845E-7, -1.617121853741662E-9, -0.0000088433540149761862},
//             {-0.10039441488979514, -0.000078081019501919665, -1.2300644175683558E-9,
//              -1.9151574911187822E-7, -3.9349062511340432E-9, -0.000016506953187581543},
//             {-0.15101734453237334, -0.00014270050135188576, 2.5577627771236027E-9,
//              -1.4305425754357427E-7, -4.5983912380703506E-9, -0.000017237934670614527},
//             {-0.19591478550134453, -0.00019646890107545141, 1.0001180311677349E-8,
//              -1.0895176284170809E-7, -6.3964066992572377E-9, -0.000016724507946719819},
//             {-0.21887783838736499, -0.00022340063093158737, 1.6975877286895753E-8,
//              -1.0449235555339682E-7, -7.1144364883295513E-9, -0.000016664004302439988},
//         }
//     );
// }

// TEST_F(SolverStep2Test, SolverUpdateStatePrediction_q) {
//     expect_kokkos_view_2D_equal(
//         solver_->state.q,
//         {
//             {-9.9999991642894645E-7, 0.0019999996666666834, -2.7693137528041648E-28,
//              0.99999987500000264, -3.2977102872969255E-30, 6.5954200250787654E-27,
//              0.0004999999791666669},
//             {-0.0000014964806473087194, 0.003174722650848854, -4.5434716051775647E-14,
//              0.99999987500005827, -1.5960341795939002E-12, -2.0361644250335121E-14,
//              0.00049999986781747402},
//             {-0.0000025281897479172786, 0.0055738404513157504, -3.0976187800764434E-14,
//              0.99999987500010656, -2.4114296289607408E-12, -4.9545531190880803E-14,
//              0.0004999997713229107},
//             {-0.0000038030054024772931, 0.0084261539888373388, 6.4411049537235008E-14,
//              0.99999987500011112, -1.8012371138628233E-12, -5.7899660620383827E-14,
//              0.0004999997621189143},
//             {-0.000004933638516780269, 0.010825271672054751, 2.5185545987603492E-13,
//              0.99999987500010789, -1.3718428393610572E-12, -8.0538988072784165E-14,
//              0.00049999976858361656},
//             {-0.0000055119072875186336, 0.011999994374197159, 4.2749628020387186E-13,
//              0.99999987500010756, -1.3156931654438724E-12, -8.9579906722424441E-14,
//              0.00049999976934543534},
//         }
//     );
// }

// TEST_F(SolverStep2Test, SolverUpdateStatePrediction_lambda) {
//     expect_kokkos_view_1D_equal(
//         solver_->state.lambda,
//         {
//             -0.10816660597819647,
//             -0.000095455157310304377,
//             2.1220160418770112E-9,
//             -2.0316113427113777E-8,
//             -2.1788291524533811E-8,
//             0.000033726153743119725,
//         }
//     );
// }

}  // namespace openturbine::restruct_poc::tests
