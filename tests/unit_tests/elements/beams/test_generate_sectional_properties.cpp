#include <gtest/gtest.h>

#include "elements/beams/generate_sectional_properties.hpp"

namespace openturbine::tests {

class GenerateSectionalMatricesTest : public ::testing::Test {
protected:
    static constexpr double kTolerance = 1e-12;

    void ExpectMatrixEqual(const Array_6x6& actual, const Array_6x6& expected) {
        for (size_t i = 0; i < 6; ++i) {
            for (size_t j = 0; j < 6; ++j) {
                EXPECT_NEAR(actual[i][j], expected[i][j], kTolerance)
                    << "Matrix element [" << i << "][" << j << "] differs";
            }
        }
    }
};

TEST_F(GenerateSectionalMatricesTest, StiffnessMatrix_Uncoupled) {
    // Test with no coupling (all offsets zero, principal angles zero)
    const double EA = 1000.;
    const double EI_x = 200.;
    const double EI_y = 300.;
    const double GKt = 150.;
    const double GA = 400.;
    const double kxs = 1.;
    const double kys = 1.;
    const double x_C = 0.;      // No centroid offset in x
    const double y_C = 0.;      // No centroid offset in y
    const double theta_p = 0.;  // No principal axis rotation
    const double x_S = 0.;      // No shear center offset in x
    const double y_S = 0.;      // No shear center offset in y
    const double theta_s = 0.;  // No shear axis rotation

    auto result = GenerateStiffnessMatrix(
        EA, EI_x, EI_y, GKt, GA, kxs, kys, x_C, y_C, theta_p, x_S, y_S, theta_s
    );

    Array_6x6 expected = {};
    expected[0][0] = GA;    // Shear stiffness in x
    expected[1][1] = GA;    // Shear stiffness in y
    expected[2][2] = EA;    // Axial stiffness
    expected[3][3] = EI_x;  // Bending stiffness about x
    expected[4][4] = EI_y;  // Bending stiffness about y
    expected[5][5] = GKt;   // Torsional stiffness

    ExpectMatrixEqual(result, expected);
}

TEST_F(GenerateSectionalMatricesTest, StiffnessMatrix_CentroidOffsetCoupling) {
    // Test axial-bending coupling due to centroid offset
    const double EA = 1000.;
    const double EI_x = 200.;
    const double EI_y = 300.;
    const double GKt = 150.;
    const double GA = 400.;
    const double kxs = 1.;
    const double kys = 1.;
    const double x_C = 0.1;     // Centroid offset in x
    const double y_C = 0.2;     // Centroid offset in y
    const double theta_p = 0.;  // No principal axis rotation
    const double x_S = 0.;      // No shear center offset in x
    const double y_S = 0.;      // No shear center offset in y
    const double theta_s = 0.;  // No shear axis rotation

    auto result = GenerateStiffnessMatrix(
        EA, EI_x, EI_y, GKt, GA, kxs, kys, x_C, y_C, theta_p, x_S, y_S, theta_s
    );

    // Expected stiffness matrix
    Array_6x6 expected = {};

    // Shear stiffness
    expected[0][0] = GA;
    expected[1][1] = GA;

    // Axial stiffness
    expected[2][2] = EA;

    // Torsional stiffness
    expected[5][5] = GKt;

    // Axial-bending coupling
    expected[2][3] = EA * y_C;
    expected[2][4] = -EA * x_C;
    expected[3][2] = EA * y_C;
    expected[4][2] = -EA * x_C;

    // Bending stiffness with axial coupling
    expected[3][3] = EI_x + EA * y_C * y_C;
    expected[4][4] = EI_y + EA * x_C * x_C;
    expected[3][4] = -EA * x_C * y_C;
    expected[4][3] = -EA * x_C * y_C;

    ExpectMatrixEqual(result, expected);
}

TEST_F(GenerateSectionalMatricesTest, MassMatrix_Uncoupled) {
    // Test basic mass matrix with no coupling (CG at origin, no principal axis rotation)
    const double m = 10.;
    const double I_x = 5.;
    const double I_y = 8.;
    const double I_p = I_x + I_y;  // 13.
    const double x_G = 0.;         // No CG offset
    const double y_G = 0.;
    const double theta_i = 0.;  // No principal axis rotation

    auto result = GenerateMassMatrix(m, I_x, I_y, I_p, x_G, y_G, theta_i);

    Array_6x6 expected = {};

    // Translational mass
    expected[0][0] = m;
    expected[1][1] = m;
    expected[2][2] = m;

    // Rotational inertia (no rotation, so principal = reference frame)
    expected[3][3] = I_x;
    expected[4][4] = I_y;
    expected[5][5] = I_p;

    ExpectMatrixEqual(result, expected);
}

TEST_F(GenerateSectionalMatricesTest, MassMatrix_CGOffsetCoupling) {
    // Test translational-rotational coupling due to CG offset
    const double m = 10.;
    const double I_x = 5.;
    const double I_y = 8.;
    const double I_p = I_x + I_y;  // 13.
    const double x_G = 0.2;        // CG offset in x
    const double y_G = 0.3;        // CG offset in y
    const double theta_i = 0.;     // No principal axis rotation

    auto result = GenerateMassMatrix(m, I_x, I_y, I_p, x_G, y_G, theta_i);

    Array_6x6 expected = {};

    // Translational mass
    expected[0][0] = m;
    expected[1][1] = m;
    expected[2][2] = m;

    // Translational-rotational coupling due to CG offset
    expected[0][5] = -m * y_G;  // F_x due to α_z
    expected[1][5] = m * x_G;   // F_y due to α_z
    expected[2][3] = m * y_G;   // F_z due to α_x
    expected[2][4] = -m * x_G;  // F_z due to α_y

    // Rotational-translational coupling due to CG offset (symmetric)
    expected[3][2] = m * y_G;   // M_x due to a_z
    expected[4][2] = -m * x_G;  // M_y due to a_z
    expected[5][0] = -m * y_G;  // M_z due to a_x
    expected[5][1] = m * x_G;   // M_z due to a_y

    // Rotational inertia with CG coupling
    expected[3][3] = I_x + m * y_G * y_G;
    expected[4][4] = I_y + m * x_G * x_G;
    expected[3][4] = -m * x_G * y_G;
    expected[4][3] = -m * x_G * y_G;

    // Polar inertia with CG coupling
    expected[5][5] = I_p + m * (x_G * x_G + y_G * y_G);

    ExpectMatrixEqual(result, expected);
}

}  // namespace openturbine::tests
