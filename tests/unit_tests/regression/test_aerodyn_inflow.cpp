#include <gtest/gtest.h>

#include "test_utilities.hpp"

#include "src/utilities/aerodynamics/aerodyn_inflow.hpp"
#include "src/vendor/dylib/dylib.hpp"

namespace openturbine::tests {

#ifdef OpenTurbine_BUILD_OPENFAST_ADI

TEST(AerodynInflowTest, ErrorHandling_NoThrow) {
    util::ErrorHandling error_handling;
    EXPECT_NO_THROW(error_handling.CheckError());
}

TEST(AerodynInflowTest, ErrorHandling_Throw) {
    util::ErrorHandling error_handling;
    error_handling.error_status = 1;
    EXPECT_THROW(error_handling.CheckError(), std::runtime_error);
}

TEST(AerodynInflowTest, FluidProperties_Default) {
    util::FluidProperties fluid_properties;
    EXPECT_NEAR(fluid_properties.density, 1.225f, 1e-6f);
    EXPECT_NEAR(fluid_properties.kinematic_viscosity, 1.464E-5f, 1e-6f);
    EXPECT_NEAR(fluid_properties.sound_speed, 335.f, 1e-6f);
    EXPECT_NEAR(fluid_properties.vapor_pressure, 1700.f, 1e-6f);
}

TEST(AerodynInflowTest, FluidProperties_Set) {
    util::FluidProperties fluid_properties;
    fluid_properties.density = 1.1f;
    fluid_properties.kinematic_viscosity = 1.5E-5f;
    fluid_properties.sound_speed = 340.f;
    fluid_properties.vapor_pressure = 1800.f;

    EXPECT_NEAR(fluid_properties.density, 1.1f, 1e-6f);
    EXPECT_NEAR(fluid_properties.kinematic_viscosity, 1.5E-5f, 1e-6f);
    EXPECT_NEAR(fluid_properties.sound_speed, 340.f, 1e-6f);
    EXPECT_NEAR(fluid_properties.vapor_pressure, 1800.f, 1e-6f);
}

TEST(AerodynInflowTest, EnvironmentalConditions_Default) {
    util::EnvironmentalConditions environmental_conditions;
    EXPECT_NEAR(environmental_conditions.gravity, 9.80665f, 1e-6f);
    EXPECT_NEAR(environmental_conditions.atm_pressure, 103500.f, 1e-6f);
    EXPECT_NEAR(environmental_conditions.water_depth, 0.f, 1e-6f);
    EXPECT_NEAR(environmental_conditions.msl_offset, 0.f, 1e-6f);
}

TEST(AerodynInflowTest, EnvironmentalConditions_Set) {
    util::EnvironmentalConditions environmental_conditions;
    environmental_conditions.gravity = 9.79665f;
    environmental_conditions.atm_pressure = 103000.f;
    environmental_conditions.water_depth = 100.f;
    environmental_conditions.msl_offset = 10.f;

    EXPECT_NEAR(environmental_conditions.gravity, 9.79665f, 1e-6f);
    EXPECT_NEAR(environmental_conditions.atm_pressure, 103000.f, 1e-6f);
    EXPECT_NEAR(environmental_conditions.water_depth, 100.f, 1e-6f);
    EXPECT_NEAR(environmental_conditions.msl_offset, 10.f, 1e-6f);
}

/// Check if members of the provided array is equal to the provided expected array
template <typename T, size_t N>
void ExpectArrayNear(
    const std::array<T, N>& actual, const std::array<T, N>& expected,
    T epsilon = static_cast<T>(1e-6)
) {
    ASSERT_EQ(actual.size(), expected.size());
    for (size_t i = 0; i < N; ++i) {
        EXPECT_NEAR(actual[i], expected[i], epsilon) << "Element mismatch at index " << i;
    }
}

TEST(AerodynInflowTest, SetPositionAndOrientation) {
    std::array<double, 7> data = {1., 2., 3., 0.707107, 0.707107, 0., 0.};
    std::array<float, 3> position;
    std::array<double, 9> orientation;

    util::SetPositionAndOrientation(data, position, orientation);

    ExpectArrayNear(position, {1.0f, 2.0f, 3.0f});
    ExpectArrayNear(orientation, {1., 0., 0., 0., 0., -1., 0., 1., 0.});
}

#endif

}  // namespace openturbine::tests