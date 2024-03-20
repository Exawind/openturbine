#include <initializer_list>

#include <gtest/gtest.h>

#include "src/restruct_poc/beams.hpp"
#include "tests/unit_tests/gen_alpha_poc/test_utilities.h"

namespace oturb::restruct_poc::tests {

std::vector<std::vector<double>> Array6x6ToVector(std::array<std::array<double, 6>, 6> A) {
    std::vector<std::vector<double>> B(6);
    for (int i = 0; i < 6; ++i) {
        B[i] = std::vector<double>(6);
        for (int j = 0; j < 6; ++j) {
            B[i][j] = A[i][j];
        }
    }
    return B;
}

TEST(BeamsTest, InitializeBeamsStruct) {
    // Stiffness matrix for uniform composite beam section
    std::array<std::array<double, 6>, 6> stiffness_matrix = {{
        {1.36817e6, 0., 0., 0., 0., 0.},      // row 1
        {0., 88560., 0., 0., 0., 0.},         // row 2
        {0., 0., 38780., 0., 0., 0.},         // row 3
        {0., 0., 0., 16960., 17610., -351.},  // row 4
        {0., 0., 0., 17610., 59120., -370.},  // row 5
        {0., 0., 0., -351., -370., 141470.}   // row 6
    }};

    // Mass matrix for uniform composite beam section
    std::array<std::array<double, 6>, 6> mass_matrix = {{
        {2., 0., 0., 0., 0.6, -0.4},  // row 1
        {0., 2., 0., -0.6, 0., 0.2},  // row 2
        {0., 0., 2., 0.4, -0.2, 0.},  // row 3
        {0., -0.6, 0.4, 1., 2., 3.},  // row 4
        {0.6, 0., -0.2, 2., 4., 6.},  // row 5
        {-0.4, 0.2, 0., 3., 6., 9.}   // row 6
    }};

    // Define beam initialization
    std::vector<BeamInput> beam_elem_inputs = {BeamInput(
        openturbine::gebt_poc::UserDefinedQuadrature(
            {
                -0.9491079123427585,  // point 1
                -0.7415311855993945,  // point 2
                -0.4058451513773972,  // point 3
                0.,                   // point 4
                0.4058451513773972,   // point 5
                0.7415311855993945,   // point 6
                0.9491079123427585    // point 7
            },
            {
                0.1294849661688697,  // weight 1
                0.2797053914892766,  // weight 2
                0.3818300505051189,  // weight 3
                0.4179591836734694,  // weight 4
                0.3818300505051189,  // weight 5
                0.2797053914892766,  // weight 6
                0.1294849661688697   // weight 7
            }
        ),
        {
            BeamNode({1., 0., 0.}, {1., 0., 0., 0.}),                  // node 1
            BeamNode({2.7267316464601146, 0., 0.}, {1., 0., 0., 0.}),  // node 2
            BeamNode({6., 0., 0.}, {1., 0., 0., 0.}),                  // node 3
            BeamNode({9.273268353539885, 0., 0.}, {1., 0., 0., 0.}),   // node 4
            BeamNode({11., 0., 0.}, {1., 0., 0., 0.})                  // node 5
        },
        {
            BeamSection(0., mass_matrix, stiffness_matrix),
            BeamSection(1., mass_matrix, stiffness_matrix),
        }
    )};

    // Initialize beams from element inputs
    auto beams = InitializeBeams(beam_elem_inputs);

    //--------------------------------------------------------------------------
    // Check that beam element views have been initialized
    //--------------------------------------------------------------------------

    // Quadrature point weights
    openturbine::gen_alpha_solver::tests::expect_kokkos_view_1D_equal(
        beams.qp_weight, beam_elem_inputs[0].quadrature.GetQuadratureWeights()
    );

    openturbine::gen_alpha_solver::tests::expect_kokkos_view_2D_equal(
        Kokkos::subview(beams.qp_Mstar, 0, Kokkos::ALL, Kokkos::ALL),
        Array6x6ToVector(beam_elem_inputs[0].sections[0].M_star)
    );

    openturbine::gen_alpha_solver::tests::expect_kokkos_view_2D_equal(
        Kokkos::subview(beams.qp_Cstar, 0, Kokkos::ALL, Kokkos::ALL),
        Array6x6ToVector(beam_elem_inputs[0].sections[0].C_star)
    );
}

}  // namespace oturb::restruct_poc::tests
