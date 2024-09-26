#include <gtest/gtest.h>

#include "test_utilities.hpp"

#include "src/utilities/aerodynamics/aerodyn_inflow.hpp"
#include "src/vendor/dylib/dylib.hpp"

namespace openturbine::tests {

#ifdef OpenTurbine_BUILD_OPENFAST_ADI
TEST(AerodynInflowTest, ADI_C_PreInit) {
    // Use dylib to load the dynamic library and get access to the aerodyn inflow c binding functions
    const std::filesystem::path project_root = FindProjectRoot();
    const std::filesystem::path full_path =
        project_root / "build/tests/unit_tests/libaerodyn_inflow_c_binding.";
    std::string path = full_path.string();
#ifdef __APPLE__
    path += "dylib";
#elif __linux__
    path += "so";
#else  // Windows
    path += "dll";
#endif
    const util::dylib lib(path, util::dylib::no_filename_decorations);
    auto ADI_C_PreInit = lib.get_function<void(int*, int*, int*, int*, char*)>("ADI_C_PreInit");

    // Call ADI_C_PreInit routine and expect the following outputs
    int numTurbines{1};               // input: Number of turbines
    int transposeDCM{1};              // input: Transpose the direction cosine matrix
    int debuglevel{0};                // input: Debug level
    int error_status_c{0};            // output: error status
    char error_message_c[] = {'\0'};  // output: error message
    ADI_C_PreInit(
        &numTurbines, &transposeDCM, &debuglevel, &error_status_c,
        static_cast<char*>(error_message_c)
    );

    EXPECT_EQ(numTurbines, 1);
    EXPECT_EQ(transposeDCM, 1);
    EXPECT_EQ(debuglevel, 0);
    EXPECT_EQ(error_status_c, 0);
    EXPECT_STREQ(static_cast<char*>(error_message_c), "");
}

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
    EXPECT_EQ(fluid_properties.density, 1.225f);
    EXPECT_EQ(fluid_properties.kinematic_viscosity, 1.464E-5f);
    EXPECT_EQ(fluid_properties.sound_speed, 335.f);
    EXPECT_EQ(fluid_properties.vapor_pressure, 1700.f);
}

TEST(AerodynInflowTest, FluidProperties_Set) {
    util::FluidProperties fluid_properties;
    fluid_properties.density = 1.1f;
    fluid_properties.kinematic_viscosity = 1.5E-5f;
    fluid_properties.sound_speed = 340.f;
    fluid_properties.vapor_pressure = 1800.f;
    EXPECT_EQ(fluid_properties.density, 1.1f);
    EXPECT_EQ(fluid_properties.kinematic_viscosity, 1.5E-5f);
    EXPECT_EQ(fluid_properties.sound_speed, 340.f);
    EXPECT_EQ(fluid_properties.vapor_pressure, 1800.f);
}

TEST(AerodynInflowTest, EnvironmentalConditions_Default) {
    util::EnvironmentalConditions environmental_conditions;
    EXPECT_EQ(environmental_conditions.gravity, 9.80665f);
    EXPECT_EQ(environmental_conditions.atm_pressure, 103500.f);
    EXPECT_EQ(environmental_conditions.water_depth, 0.f);
    EXPECT_EQ(environmental_conditions.msl_offset, 0.f);
}

TEST(AerodynInflowTest, EnvironmentalConditions_Set) {
    util::EnvironmentalConditions environmental_conditions;
    environmental_conditions.gravity = 9.79665f;
    environmental_conditions.atm_pressure = 103000.f;
    environmental_conditions.water_depth = 100.f;
    environmental_conditions.msl_offset = 10.f;
    EXPECT_EQ(environmental_conditions.gravity, 9.79665f);
    EXPECT_EQ(environmental_conditions.atm_pressure, 103000.f);
    EXPECT_EQ(environmental_conditions.water_depth, 100.f);
    EXPECT_EQ(environmental_conditions.msl_offset, 10.f);
}

#endif

}  // namespace openturbine::tests