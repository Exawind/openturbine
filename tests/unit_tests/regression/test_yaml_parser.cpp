#include <gtest/gtest.h>
#include <yaml-cpp/yaml.h>

#include "test_utilities.hpp"

#include "src/utilities/scripts/windio_mapped_structs.hpp"

namespace openturbine::tests {

TEST(ParserTest, ParseIEA15MWBasicInfo) {
    const std::filesystem::path projectRoot = FindProjectRoot();
    const std::filesystem::path yamlPath = projectRoot / "src/utilities/scripts/IEA-15-240-RWT.yaml";
    const YAML::Node config = YAML::LoadFile(yamlPath.string());
    Turbine turbine;
    turbine.parse(config);
    ASSERT_EQ(turbine.name, "IEA 15MW Offshore Reference Turbine, with taped chord tip design");
}

TEST(ParserTest, ParseIEA15MWAssembly) {
    const std::filesystem::path projectRoot = FindProjectRoot();
    const std::filesystem::path yamlPath = projectRoot / "src/utilities/scripts/IEA-15-240-RWT.yaml";
    const YAML::Node config = YAML::LoadFile(yamlPath.string());
    Turbine turbine;
    turbine.parse(config);
    ASSERT_EQ(turbine.assembly.turbine_class, "I");
    ASSERT_EQ(turbine.assembly.turbulence_class, "B");
    ASSERT_EQ(turbine.assembly.drivetrain, "direct_drive");
    ASSERT_EQ(turbine.assembly.rotor_orientation, "Upwind");
    ASSERT_EQ(turbine.assembly.number_of_blades, 3);
    ASSERT_EQ(turbine.assembly.hub_height, 150.);
    ASSERT_EQ(turbine.assembly.rotor_diameter, 241.94);
    ASSERT_EQ(turbine.assembly.rated_power, 15.e+6);
    ASSERT_EQ(turbine.assembly.lifetime, 25.);
}

TEST(ParserTest, ParseIEA15MWMaterials) {
    const std::filesystem::path projectRoot = FindProjectRoot();
    const std::filesystem::path yamlPath = projectRoot / "src/utilities/scripts/IEA-15-240-RWT.yaml";
    const YAML::Node config = YAML::LoadFile(yamlPath.string());
    Turbine turbine;
    turbine.parse(config);
    ASSERT_EQ(turbine.materials.size(), 11);
}

}  // namespace openturbine::tests
