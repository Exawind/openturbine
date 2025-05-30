#include <gtest/gtest.h>

#include "elements/beams/hollow_circle_properties.hpp"

namespace openturbine::tests {

class HollowCirclePropertiesTest : public ::testing::Test {
protected:
    static constexpr double kTolerance = 1e-12;

    struct TestParameters {
        // Geometric properties
        double outer_diameter{1.};
        double wall_thickness{0.1};
        double nu{0.33};

        // Material properties
        double E{200.E9};   // Steel Young's modulus (Pa)
        double G{80.E9};    // Steel shear modulus (Pa)
        double rho{7850.};  // Steel density (kg/m³)

        // Offset parameters
        double x_C{0.};
        double y_C{0.};
        double theta_p{0.};
        double x_S{0.};
        double y_S{0.};
        double theta_s{0.};
        double x_G{0.};
        double y_G{0.};
        double theta_i{0.};
    };

    TestParameters tp_;

    void ExpectMatrixEqual(const Array_6x6& actual, const Array_6x6& expected) {
        for (size_t i = 0; i < 6; ++i) {
            for (size_t j = 0; j < 6; ++j) {
                EXPECT_NEAR(actual[i][j], expected[i][j], kTolerance)
                    << "Matrix element [" << i << "][" << j << "] differs";
            }
        }
    }
};

TEST_F(HollowCirclePropertiesTest, CalculateGeometricProperties) {
    // Test basic geometric property calculations
    auto properties =
        CalculateHollowCircleProperties(tp_.outer_diameter, tp_.wall_thickness, tp_.nu);

    // Calculate expected values manually
    const double outer_radius = tp_.outer_diameter / 2.;
    const double inner_radius = outer_radius - tp_.wall_thickness;

    const double expected_area = M_PI * (std::pow(outer_radius, 2) - std::pow(inner_radius, 2));
    const double expected_Ixx = M_PI * (std::pow(outer_radius, 4) - std::pow(inner_radius, 4)) / 4.;
    const double expected_J = M_PI * (std::pow(outer_radius, 4) - std::pow(inner_radius, 4)) / 2.;
    const double expected_kx = (6. * (1. + tp_.nu)) / (7. + 6. * tp_.nu);

    EXPECT_NEAR(properties.area, expected_area, kTolerance);
    EXPECT_NEAR(properties.Ixx, expected_Ixx, kTolerance);
    EXPECT_NEAR(properties.Iyy, expected_Ixx, kTolerance);  // Circular symmetry
    EXPECT_NEAR(properties.J, expected_J, kTolerance);
    EXPECT_NEAR(properties.kx, expected_kx, kTolerance);
    EXPECT_NEAR(properties.ky, expected_kx, kTolerance);  // Circular symmetry
}

}  // namespace openturbine::tests
