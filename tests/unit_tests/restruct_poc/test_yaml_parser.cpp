#include <gtest/gtest.h>
#include <yaml-cpp/yaml.h>

#include "test_utilities.hpp"

#include "src/utilities/scripts/windio_mapped_structs.cpp"

namespace openturbine::restruct_poc::tests {

TEST(ParserTest, ParseIEA15MW) {
    try {
        // Load the YAML file
        YAML::Node config = YAML::LoadFile("../src/utilities/scripts/IEA-15-240-RWT.yaml");

        // Parse the top-level node
        Turbine turbine;
        turbine.parse(config);

        // Assert the turbine name
        ASSERT_EQ(turbine.name, "IEA 15MW Offshore Reference Turbine, with taped chord tip design");

        // Assert the assembly parameters
        ASSERT_EQ(turbine.assembly.turbine_class, "I");
        ASSERT_EQ(turbine.assembly.turbulence_class, "B");
        ASSERT_EQ(turbine.assembly.drivetrain, "direct_drive");
        ASSERT_EQ(turbine.assembly.rotor_orientation, "Upwind");
        ASSERT_EQ(turbine.assembly.number_of_blades, 3);
        ASSERT_EQ(turbine.assembly.hub_height, 150.);
        ASSERT_EQ(turbine.assembly.rotor_diameter, 241.94);
        ASSERT_EQ(turbine.assembly.rated_power, 15.e+6);
        ASSERT_EQ(turbine.assembly.lifetime, 25.);

        // Assert the materials
        ASSERT_EQ(turbine.materials.size(), 11);
    } catch (const YAML::Exception& e) {
        std::cerr << "Error loading YAML file: " << e.what() << std::endl;
    }
}

}  // namespace openturbine::restruct_poc::tests
