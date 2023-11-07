#include <gtest/gtest.h>

#include "src/gebt_poc/solver.h"
#include "tests/unit_tests/gen_alpha_poc/test_utilities.h"

namespace openturbine::gebt_poc::tests {

TEST(StaticCompositeBeamTest, StaticAnalysisWithZeroForceAndZeroInitialGuess) {
    // 2 nodes on a straight line
    auto position_vectors = gen_alpha_solver::create_vector({
        0., 0., 0., 1., 0., 0., 0.,  // node 1
        1., 0., 0., 1., 0., 0., 0.,  // node 2
    });

    // zero displacement and rotation as initial guess
    auto generalized_coords = gen_alpha_solver::create_vector({
        0., 0., 0., 0., 0., 0., 0.,  // node 1
        0., 0., 0., 0., 0., 0., 0.,  // node 2
    });

    // TODO Constraints for the cantilever beam with applied zero force
    // auto constraints = gen_alpha_solver::create_vector({
    //     0., 0., 0., 0., 0., 0., 0.,  // node 1
    //     0., 0., 0., 0., 0., 0., 0.,  // node 2
    // });

    // Stiffness matrix for uniform composite beam section
    auto stiffness = gen_alpha_solver::create_matrix({
        {1368.17, 0., 0., 0., 0., 0.},        // row 1
        {0., 88.56, 0., 0., 0., 0.},          // row 2
        {0., 0., 38.78, 0., 0., 0.},          // row 3
        {0., 0., 0., 16.96, 17.61, -0.351},   // row 4
        {0., 0., 0., 17.61, 59.12, -0.370},   // row 5
        {0., 0., 0., -0.351, -0.370, 141.47}  // row 6
    });

    // 7-point Gauss-Legendre quadrature for integration
    auto quadrature = UserDefinedQuadrature(
        {-0.9491079123427585, -0.7415311855993945, -0.4058451513773972, 0., 0.4058451513773972,
         0.7415311855993945, 0.9491079123427585},
        {0.1294849661688697, 0.2797053914892766, 0.3818300505051189, 0.4179591836734694,
         0.3818300505051189, 0.2797053914892766, 0.1294849661688697}
    );

    // TODO Perform static analysis via generalized-alpha time integration
}

}  // namespace openturbine::gebt_poc::tests
