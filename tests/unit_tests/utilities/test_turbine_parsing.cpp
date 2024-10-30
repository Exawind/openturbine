#include <gtest/gtest.h>
#include <yaml-cpp/yaml.h>

#include "src/utilities/scripts/windio_mapped_structs.hpp"

namespace openturbine::tests {

TEST(ParserTest, ParseIEA15MW_Name) {
    const std::string input = R"(
        name: IEA 15MW Offshore Reference Turbine, with taped chord tip design
    )";
    const YAML::Node config = YAML::Load(input);
    Turbine turbine;
    turbine.parse(config);

    ASSERT_EQ(turbine.name, "IEA 15MW Offshore Reference Turbine, with taped chord tip design");
}

TEST(ParserTest, ParseIEA15MW_Assembly) {
    const std::string input = R"(
        assembly:
            turbine_class: I
            turbulence_class: B
            drivetrain: direct_drive
            rotor_orientation: Upwind
            number_of_blades: 3
            hub_height: 150
            rotor_diameter: 241.94
            rated_power: 15.e+6
            lifetime: 25
    )";
    const YAML::Node config = YAML::Load(input);
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

TEST(ParserTest, ParseIEA15MW_MultipleMaterials) {
    const std::string input = R"(
        materials:
            - name: first
            - name: second
        )";
    const YAML::Node config = YAML::Load(input);
    Turbine turbine;
    turbine.parse(config);

    EXPECT_EQ(turbine.materials.size(), 2);
    EXPECT_EQ(turbine.materials[0].name, "first");
    EXPECT_EQ(turbine.materials[1].name, "second");
}

}  // namespace openturbine::tests
