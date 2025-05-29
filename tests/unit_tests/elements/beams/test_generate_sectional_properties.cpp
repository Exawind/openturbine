#include <gtest/gtest.h>

#include "elements/beams/generate_sectional_properties.hpp"

namespace openturbine::tests {

class GenerateStiffnessMatrixTest : public ::testing::Test {
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

TEST_F(GenerateStiffnessMatrixTest, Uncoupled) {
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

TEST_F(GenerateStiffnessMatrixTest, CentroidOffsetCoupling) {
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

}  // namespace openturbine::tests
